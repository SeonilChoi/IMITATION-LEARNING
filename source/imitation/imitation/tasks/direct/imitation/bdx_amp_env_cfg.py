import os

from imitation.robots import BDX_CFG, ISAACLAB_ASSETS_DIR

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class BdxAmpEnvCfg(DirectRLEnvCfg):
    # environment configuration
    decimation = 2
    episode_length_s = 10.0

    # space configuration
    action_space = 16  # 16D vector of joint positions
    observation_space = 131  # 131D vector of robot state
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = 131  # 131D vector of robot state
    action_scale = 1.0

    early_termination = True
    termination_height = 0.2
    reference_body = "pelvis"
    reset_strategy = "random"

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            bounce_threshold_velocity=0.2,
        ),
    )

    # robot
    robot: ArticulationCfg = BDX_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # terrain
    terrain: TerrainImporterCfg = TerrainImporterCfg(
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192,
        env_spacing=4.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # custom parameters
    dt = 1 / 120 * 2
    motion_folder_path = os.path.join(ISAACLAB_ASSETS_DIR, "motions", "bdx")
