from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

import isaaclab.sim as sim_utils

from imitation.robots import ISAACLAB_ASSETS_DIR


JOINT_INIT_POS = {
    "left_ankle": -0.02488,
    "left_knee": -0.03127,
    "left_hip_pitch": -0.00640,
    "left_hip_roll": -0.03041,
    "left_hip_yaw": 0.0,
    "left_antenna": 0.0,
    "right_ankle": 0.02477,
    "right_knee": 0.03099,
    "right_hip_pitch": 0.00622,
    "right_hip_roll": 0.03081,
    "right_hip_yaw": 0.00000,
    "right_antenna": 0.0,
    "neck_pitch": 0.0,
    "head_pitch": 0.0,
    "head_roll": 0.0,
    "head_yaw": 0.0,
}

JOINT_STIFFNESS = {
    "left_ankle": 30.0,
    "left_knee": 35.0,
    "left_hip_pitch": 40.0,
    "left_hip_roll": 40.0,
    "left_hip_yaw": 40.0,
    "left_antenna": 3.0,
    "right_ankle": 30.0,
    "right_knee": 35.0,
    "right_hip_pitch": 40.0,
    "right_hip_roll": 40.0,
    "right_hip_yaw": 40.0,
    "right_antenna": 3.0,
    "neck_pitch": 10.0,
    "head_pitch": 5.0,
    "head_roll": 5.0,
    "head_yaw": 5.0,
}

JOINT_DAMPING = {
    "left_ankle": 1.5,
    "left_knee": 1.5,
    "left_hip_pitch": 1.5,
    "left_hip_roll": 1.5,
    "left_hip_yaw": 1.5,
    "left_antenna": 1.5,
    "right_ankle": 1.5,
    "right_knee": 1.5,
    "right_hip_pitch": 1.5,
    "right_hip_roll": 1.5,
    "right_hip_yaw": 1.5,
    "right_antenna": 1.5,
    "neck_pitch": 1.5,
    "head_pitch": 1.5,
    "head_roll": 1.5,
    "head_yaw": 1.5,
}

JOINT_EFFORT_LIMIT = {
    "left_ankle": 100.0,
    "left_knee": 100.0,
    "left_hip_pitch": 100.0,
    "left_hip_roll": 100.0,
    "left_hip_yaw": 100.0,
    "left_antenna": 10.0,
    "right_ankle": 100.0,
    "right_knee": 100.0,
    "right_hip_pitch": 100.0,
    "right_hip_roll": 100.0,
    "right_hip_yaw": 100.0,
    "right_antenna": 10.0,
    "neck_pitch": 50.0,
    "head_pitch": 50.0,
    "head_roll": 50.0,
    "head_yaw": 50.0,
}

JOINT_VELOCITY_LIMIT = {
    "left_ankle": 10.0,
    "left_knee": 10.0,
    "left_hip_pitch": 10.0,
    "left_hip_roll": 10.0,
    "left_hip_yaw": 10.0,
    "left_antenna": 10.0,
    "right_ankle": 10.0,
    "right_knee": 10.0,
    "right_hip_pitch": 10.0,
    "right_hip_roll": 10.0,
    "right_hip_yaw": 10.0,
    "right_antenna": 10.0,
    "neck_pitch": 10.0,
    "head_pitch": 10.0,
    "head_roll": 10.0,
    "head_yaw": 10.0,
}


BDX_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DIR}/models/bdx/bdx/bdx.usd",
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
