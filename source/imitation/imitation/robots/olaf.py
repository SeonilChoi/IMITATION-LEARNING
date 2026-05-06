from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

import isaaclab.sim as sim_utils

from imitation.robots import ISAACLAB_ASSETS_DIR


JOINT_INIT_POS = {
    "left_ankle_pitch": 0.0,
    "left_ankle_roll": 0.0,
    "left_knee": 0.0,
    "left_hip_pitch": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_yaw": 0.0,
    "right_ankle_pitch": 0.0,
    "right_ankle_roll": 0.0,
    "right_knee": 0.0,
    "right_hip_pitch": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_yaw": 0.0,
}

JOINT_STIFFNESS = {
    "left_ankle_pitch": 0.0,
    "left_ankle_roll": 0.0,
    "left_knee": 0.0,
    "left_hip_pitch": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_yaw": 0.0,
    "right_ankle_pitch": 0.0,
    "right_ankle_roll": 0.0,
    "right_knee": 0.0,
    "right_hip_pitch": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_yaw": 0.0,
}

JOINT_DAMPING = {
    "left_ankle_pitch": 0.0,
    "left_ankle_roll": 0.0,
    "left_knee": 0.0,
    "left_hip_pitch": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_yaw": 0.0,
    "right_ankle_pitch": 0.0,
    "right_ankle_roll": 0.0,
    "right_knee": 0.0,
    "right_hip_pitch": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_yaw": 0.0, 
}

JOINT_EFFORT_LIMIT = {
    "left_ankle_pitch": 100.0,
    "left_ankle_roll": 100.0,
    "left_knee": 100.0,
    "left_hip_pitch": 100.0,
    "left_hip_roll": 100.0,
    "left_hip_yaw": 100.0,
    "right_ankle_pitch": 100.0,
    "right_ankle_roll": 100.0,
    "right_knee": 100.0,
    "right_hip_pitch": 100.0,
    "right_hip_roll": 100.0,
    "right_hip_yaw": 100.0,
}

JOINT_VELOCITY_LIMIT = {
    "left_ankle_pitch": 30.0,
    "left_ankle_roll": 30.0,
    "left_knee": 30.0,
    "left_hip_pitch": 30.0,
    "left_hip_roll": 30.0,
    "left_hip_yaw": 30.0,
    "right_ankle_pitch": 30.0,
    "right_ankle_roll": 30.0,
    "right_knee": 30.0,
    "right_hip_pitch": 30.0,
    "right_hip_roll": 30.0,
    "right_hip_yaw": 30.0,
}


OLAF_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DIR}/models/olaf/olaf/olaf.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos=JOINT_INIT_POS,
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=JOINT_STIFFNESS,
            damping=JOINT_DAMPING,
            effort_limit=JOINT_EFFORT_LIMIT,
            velocity_limit=JOINT_VELOCITY_LIMIT,
        ),
    },
)