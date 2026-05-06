from __future__ import annotations

import os
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from imitation.tasks.utils import MotionLoader

from .bdx_amp_env_cfg import BdxAmpEnvCfg


class BdxAmpEnv(DirectRLEnv):
    cfg: BdxAmpEnvCfg

    def __init__(self, cfg: BdxAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        motion_files = [
            os.path.join(self.cfg.motion_folder_path, f)
            for f in os.listdir(self.cfg.motion_folder_path)
            if f.endswith(".json")
        ]
        self.motion_loader = MotionLoader("bdx", motion_files, self.step_dt, self.device)

        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.left_toe_body_index = self.robot.data.body_names.index("left_foot")
        self.right_toe_body_index = self.robot.data.body_names.index("right_foot")
        self.trunk_body_index = self.robot.data.body_names.index("trunk")

        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space),
            dtype=torch.float32,
            device=self.device,
        )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        target = self.robot.data.default_joint_pos.clone()
        target = target + self.cfg.action_scale * self.actions

        self.robot.set_joint_position_target(target)

    def _get_observations(self) -> dict:
        root_position = self.robot.data.body_pos_w[:, self.ref_body_index]
        root_orientation_quat_wxyz = self.robot.data.body_quat_w[:, self.ref_body_index]
        root_orientation_quat_xyzw = root_orientation_quat_wxyz[:, [1, 2, 3, 0]]
        world_linear_velocity = self.robot.data.body_lin_vel_w[:, self.ref_body_index]
        world_angular_velocity = self.robot.data.body_ang_vel_w[:, self.ref_body_index]

        left_toe_position = self.robot.data.body_pos_w[:, self.left_toe_body_index]
        right_toe_position = self.robot.data.body_pos_w[:, self.right_toe_body_index]

        (
            root_orientation_quat_normalized_heading,
            root_linear_velocity_normalized_heading,
            local_root_angular_velocity_normalized_heading,
        ) = self.motion_loader.normalize_heading_observation(
            root_orientation_quat_xyzw,
            world_linear_velocity,
            world_angular_velocity,
        )

        joints_positions_vectorized = self.motion_loader.vectorize_joint_positions(self.robot.data.joint_pos)
        amp_obs = compute_obs(
            root_position[:, 2:3],
            root_orientation_quat_normalized_heading,
            root_linear_velocity_normalized_heading,
            local_root_angular_velocity_normalized_heading,
            joints_positions_vectorized,
            self.robot.data.joint_vel,
            left_toe_position,
            right_toe_position,
        )

        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = amp_obs
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        return {"policy": amp_obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        trunk_z = self.robot.data.body_pos_w[:, self.trunk_body_index, 2]

        if self.cfg.early_termination:
            died = trunk_z < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        if self.cfg.reset_strategy.startswith("random"):
            random_root_state, random_joint_pos, random_joint_vel, motion_ids, times = self._sample_reference_state(
                env_ids, start="start" in self.cfg.reset_strategy
            )
            root_state = random_root_state
            joint_pos = random_joint_pos
            joint_vel = random_joint_vel
        else:
            motion_ids = None
            times = np.zeros(len(env_ids), dtype=np.float64)

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        amp_observations = self.collect_reference_motions(len(env_ids), times, motion_ids)
        self.amp_observation_buffer[env_ids] = amp_observations.view(len(env_ids), self.cfg.num_amp_observations, -1)

    def _sample_reference_state(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        num_samples = len(env_ids)
        motion_ids = self.motion_loader.sample_motion_ids(num_samples)
        times = np.zeros(num_samples, dtype=np.float64) if start else self.motion_loader.sample_times(motion_ids)
        (
            root_position,
            root_orientation_quat_xyzw,
            root_linear_velocity,
            root_angular_velocity,
            joint_pos,
            joint_vel,
            _,
            _,
        ) = self.motion_loader.sample(num_samples, motion_ids=motion_ids, times=times, return_full_state=True)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = root_position + self.scene.env_origins[env_ids]
        root_state[:, 3:7] = root_orientation_quat_xyzw[:, [3, 0, 1, 2]]
        root_state[:, 7:10] = root_linear_velocity
        root_state[:, 10:13] = root_angular_velocity

        return root_state, joint_pos, joint_vel, motion_ids, times

    def collect_reference_motions(
        self, num_samples: int, current_times: np.ndarray | None = None, motion_ids: np.ndarray | None = None
    ) -> torch.Tensor:
        if motion_ids is None:
            motion_ids = self.motion_loader.sample_motion_ids(num_samples)

        if current_times is None:
            current_times = self.motion_loader.sample_times(motion_ids)

        history_times = (
            np.expand_dims(current_times, axis=-1) - self.step_dt * np.arange(0, self.cfg.num_amp_observations)
        ).reshape(-1)
        history_motion_ids = np.repeat(motion_ids, self.cfg.num_amp_observations)

        amp_observation = compute_obs(
            *self.motion_loader.sample(
                history_motion_ids.shape[0],
                motion_ids=history_motion_ids,
                times=history_times,
            )
        )
        return amp_observation.view(num_samples, -1)


@torch.jit.script
def compute_obs(
    root_height: torch.Tensor,
    root_orientation_quat_normalized_heading: torch.Tensor,
    root_linear_velocity_normalized_heading: torch.Tensor,
    local_root_angular_velocity_normalized_heading: torch.Tensor,
    joints_positions_vectorized: torch.Tensor,
    joints_velocities: torch.Tensor,
    left_toe_position: torch.Tensor,
    right_toe_position: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (
            root_height,
            root_orientation_quat_normalized_heading,
            root_linear_velocity_normalized_heading,
            local_root_angular_velocity_normalized_heading,
            joints_positions_vectorized,
            joints_velocities,
            left_toe_position,
            right_toe_position,
        ),
        dim=-1,
    )
