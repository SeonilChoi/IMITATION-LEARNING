import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a demonstration motion.")
parser.add_argument("--index", type=int, default=0, help="Index of the demo JSON to replay.")
parser.add_argument("--loop", action="store_true", default=False, help="Loop the selected demo.")
parser.add_argument("--robot", type=str, default="bdx", choices=("bdx", "olaf"), help="Robot to use.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import json
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from imitation.robots import BDX_CFG, OLAF_CFG, ISAACLAB_ASSETS_DIR


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    """Single-environment scene for demonstration replay."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot = BDX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 10.0)),
    )


def _get_robot_cfg(robot_name: str):
    if robot_name == "bdx":
        return BDX_CFG
    if robot_name == "olaf":
        return OLAF_CFG
    raise ValueError(f"Unsupported robot: {robot_name}")


def _load_demo(robot_name: str, index: int) -> dict:
    demo_file_path = os.path.join(ISAACLAB_ASSETS_DIR, "motions", robot_name, f"{index}.json")
    if not os.path.exists(demo_file_path):
        raise FileNotFoundError(f"Demo motion not found: {demo_file_path}")

    with open(demo_file_path, encoding="utf-8") as f:
        return json.load(f)


def _slice_field(frames: torch.Tensor, offsets: dict[str, int], sizes: dict[str, int], name: str) -> torch.Tensor:
    start = offsets[name]
    size = sizes[name]
    return frames[:, start : start + size]


def _xyzw_to_wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    return quat_xyzw[:, [3, 0, 1, 2]]


def _get_joint_indices(demo_joint_names: list[str], robot_joint_names: list[str]) -> list[int]:
    missing_joint_names = [name for name in robot_joint_names if name not in demo_joint_names]
    if missing_joint_names:
        missing = ", ".join(missing_joint_names)
        raise ValueError(f"Demo motion is missing robot joints: {missing}")
    return [demo_joint_names.index(name) for name in robot_joint_names]


def main():
    episode = _load_demo(args_cli.robot, args_cli.index)

    sim_cfg = sim_utils.SimulationCfg(dt=1 / 120, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 2.5, 1.5], target=[0.0, 0.0, 0.6])

    scene_cfg = ReplaySceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)
    scene_cfg.robot = _get_robot_cfg(args_cli.robot).replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(scene_cfg)
    robot = scene.articulations["robot"]

    sim.reset()
    scene.reset()

    frame_offsets = episode["frame_offset"][0]
    frame_sizes = episode["frame_size"][0]
    frame_duration = float(episode.get("frame_duration", 1.0 / max(float(episode.get("fps", 60.0)), 1.0)))
    demo_joint_names = episode["joints"]
    frames = torch.tensor(episode["frames"], dtype=torch.float32, device=sim.device)
    env_origin = scene.env_origins[0:1]
    joint_indices = _get_joint_indices(demo_joint_names, robot.data.joint_names)

    sim_dt = sim.get_physics_dt()
    steps_per_frame = max(1, round(frame_duration / sim_dt))
    frame_index = 0
    hold_count = 0

    print(f"[INFO]: Replaying {args_cli.robot} demo {args_cli.index}.json")
    print(f"[INFO]: Number of frames: {frames.shape[0]}")
    print(f"[INFO]: Frame duration: {frame_duration:.6f} s ({steps_per_frame} sim steps per frame)")

    while simulation_app.is_running():
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step()
            continue

        with torch.inference_mode():
            frame = frames[frame_index : frame_index + 1]

            root_position = _slice_field(frame, frame_offsets, frame_sizes, "root_position") + env_origin
            root_orientation = _xyzw_to_wxyz(
                _slice_field(frame, frame_offsets, frame_sizes, "root_orientation_quat")
            )
            root_linear_velocity = _slice_field(frame, frame_offsets, frame_sizes, "world_linear_velocity")
            root_angular_velocity = _slice_field(frame, frame_offsets, frame_sizes, "world_angular_velocity")
            joint_pos = _slice_field(frame, frame_offsets, frame_sizes, "joints_positions")[:, joint_indices]
            joint_vel = _slice_field(frame, frame_offsets, frame_sizes, "joints_velocities")[:, joint_indices]

            root_pose = torch.cat((root_position, root_orientation), dim=-1)
            root_velocity = torch.cat((root_linear_velocity, root_angular_velocity), dim=-1)

            robot.write_root_pose_to_sim(root_pose)
            robot.write_root_velocity_to_sim(root_velocity)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            sim.step()
            scene.update(sim_dt)

        hold_count += 1
        if hold_count >= steps_per_frame:
            hold_count = 0
            frame_index += 1
            if frame_index >= frames.shape[0]:
                if args_cli.loop:
                    frame_index = 0
                else:
                    break


if __name__ == "__main__":
    main()
    simulation_app.close()
