from __future__ import annotations

import json
import torch
import numpy as np
import xml.etree.ElementTree as ET

from pathlib import Path


class MotionLoader:
    def __init__(self, robot_name: str, motion_files: list[str], dt: float, device: torch.device):
        self._robot_name = robot_name
        self._motion_files = sorted(motion_files)
        self._dt = dt
        self._device = device

        # Load motion data
        self._data: list[dict] = []
        for motion_file in self._motion_files:
            with open(motion_file, encoding="utf-8") as f:
                self._data.append(json.load(f))

        # Calculate motion properties
        self._n_motions = len(self._data)
        self._motion_weights = np.asarray([motion.get("motion_weight", 1.0) for motion in self._data], dtype=np.float64)
        self._motion_probs = self._motion_weights / self._motion_weights.sum()
        self._n_motion_frames = np.asarray([len(motion["frames"]) for motion in self._data], dtype=np.int64)
        self._frame_durations = np.asarray([motion.get("frame_duration", self._dt) for motion in self._data], dtype=np.float64)
        self._motion_durations = self._frame_durations * np.maximum(self._n_motion_frames - 1, 0)

        # Get motion properties from first motion
        temp_motion_data = self._data[0]
        self._joint_names = temp_motion_data["joints"]
        self._frame_offsets = temp_motion_data["frame_offset"][0]
        self._frame_sizes = temp_motion_data["frame_size"][0]

        # Load URDF file and get joint axes
        urdf_path = Path(__file__).resolve().parents[2] / "assets" / "models" / f"{self._robot_name}" / f"{self._robot_name}.urdf"
        urdf_root = ET.parse(urdf_path).getroot()
        joint_axis_map = {}
        for joint in urdf_root.findall("joint"):
            axis = joint.find("axis")
            if axis is not None:
                joint_axis_map[joint.attrib["name"]] = [float(v) for v in axis.attrib["xyz"].split()]
        self._joint_axes = torch.tensor(
            [joint_axis_map[name] for name in self._joint_names],
            dtype=torch.float32,
            device=self._device,
        )


    def interpolate(self, a: torch.Tensor, *,
        b: torch.Tensor | None = None,
        blend: torch.Tensor | None = None,
        start: np.ndarray | None = None,
        end: np.ndarray | None = None
    ) -> torch.Tensor:
        """Linear interpolation between consecutive values."""
        if start is not None and end is not None:
            return self.interpolate(a=a[start], b=a[end], blend=blend)
        if a.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if a.ndim >= 3:
            blend = blend.unsqueeze(-1)
        return (1.0 - blend) * a + blend * b

    def slerp(self, q0: torch.Tensor, *,
        q1: torch.Tensor | None = None,
        blend: torch.Tensor | None = None,
        start: np.ndarray | None = None,
        end: np.ndarray | None = None
    ) -> torch.Tensor:
        """Quaternion interpolation between consecutive rotations."""
        if start is not None and end is not None:
            return self.slerp(q0=q0[start], q1=q0[end], blend=blend)
        if q0.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if q0.ndim >= 3:
            blend = blend.unsqueeze(-1)

        qw, qx, qy, qz = 0, 1, 2, 3
        cos_half_theta = (
            q0[..., qw] * q1[..., qw] +
            q0[..., qx] * q1[..., qx] +
            q0[..., qy] * q1[..., qy] +
            q0[..., qz] * q1[..., qz]
        )

        neg_mask = cos_half_theta < 0
        q1 = q1.clone()
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta).unsqueeze(-1)

        half_theta = torch.acos(cos_half_theta)
        sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

        ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
        ratio_b = torch.sin(blend * half_theta) / sin_half_theta

        new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
        new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
        new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
        new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

        new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1)
        new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
        new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
        return new_q

    def compute_frame_blend(self, motion_ids: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert motion times into frame indexes and blend coefficients."""
        duration = self._motion_durations[motion_ids]
        frame_duration = self._frame_durations[motion_ids]
        n_motion_frames = self._n_motion_frames[motion_ids]

        phase = np.clip(times / np.maximum(duration, frame_duration), 0.0, 1.0)
        index_0 = np.floor(phase * np.maximum(n_motion_frames - 1, 0)).astype(np.int64)
        index_1 = np.minimum(index_0 + 1, n_motion_frames - 1)
        blend = np.clip((times - index_0 * frame_duration) / frame_duration, 0.0, 1.0)
        return index_0, index_1, blend

    def normalize_heading_observation(self, root_orientation_quat: torch.Tensor, root_linear_velocity: torch.Tensor, local_root_angular_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Remove root yaw heading from orientation and velocity observations."""
        
        def normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
            norm = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps)
            return q / norm

        def quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
            x1, y1, z1, w1 = q.unbind(dim=-1)
            x2, y2, z2, w2 = r.unbind(dim=-1)

            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            return torch.stack((x, y, z, w), dim=-1)

        def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            q_xyz = q[..., :3]
            q_w = q[..., 3:4]
            t = 2.0 * torch.cross(q_xyz, v, dim=-1)
            return v + q_w * t + torch.cross(q_xyz, t, dim=-1)

        def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
            axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(1e-8)
            half_angle = angle * 0.5
            sin_half = torch.sin(half_angle).unsqueeze(-1)
            cos_half = torch.cos(half_angle).unsqueeze(-1)
            xyz = axis * sin_half
            return torch.cat((xyz, cos_half), dim=-1)

        def calc_heading(q: torch.Tensor) -> torch.Tensor:
            ref_dir = torch.zeros_like(q[..., :3])
            ref_dir[..., 0] = 1.0
            rot_dir = quat_rotate(q, ref_dir)
            return torch.atan2(rot_dir[..., 1], rot_dir[..., 0])

        def calc_heading_quat_inv(q: torch.Tensor) -> torch.Tensor:
            heading = calc_heading(q)
            axis = torch.zeros_like(q[..., :3])
            axis[..., 2] = 1.0
            return quat_from_angle_axis(-heading, axis)

        def quat_to_tan_norm(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
            q = normalize_quaternion(q, eps)

            ref_tan = torch.zeros_like(q[..., :3])
            ref_tan[..., 0] = 1.0
            tan = quat_rotate(q, ref_tan)

            ref_norm = torch.zeros_like(q[..., :3])
            ref_norm[..., 2] = 1.0
            norm = quat_rotate(q, ref_norm)

            return torch.cat((tan, norm), dim=-1)

        root_orientation_quat = normalize_quaternion(root_orientation_quat)
        heading_rot = calc_heading_quat_inv(root_orientation_quat)
        heading_free_rot = quat_mul(heading_rot, root_orientation_quat)

        root_orientation_quat_normalized_heading = quat_to_tan_norm(heading_free_rot)
        root_linear_velocity_normalized_heading = quat_rotate(heading_rot, root_linear_velocity)
        local_root_angular_velocity_normalized_heading = quat_rotate(heading_rot, local_root_angular_velocity)

        return (
            root_orientation_quat_normalized_heading,
            root_linear_velocity_normalized_heading,
            local_root_angular_velocity_normalized_heading,
        )

    def vectorize_joint_positions(self, joint_positions: torch.Tensor) -> torch.Tensor:
        """Convert per-joint angles into 6D tangent-normal vectors."""

        def normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
            norm = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps)
            return q / norm

        def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            q_xyz = q[..., :3]
            q_w = q[..., 3:4]
            t = 2.0 * torch.cross(q_xyz, v, dim=-1)
            return v + q_w * t + torch.cross(q_xyz, t, dim=-1)

        def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
            axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(1e-8)
            half_angle = angle * 0.5
            sin_half = torch.sin(half_angle).unsqueeze(-1)
            cos_half = torch.cos(half_angle).unsqueeze(-1)
            xyz = axis * sin_half
            return torch.cat((xyz, cos_half), dim=-1)

        def quat_to_tan_norm(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
            q = normalize_quaternion(q, eps)

            ref_tan = torch.zeros_like(q[..., :3])
            ref_tan[..., 0] = 1.0
            tan = quat_rotate(q, ref_tan)

            ref_norm = torch.zeros_like(q[..., :3])
            ref_norm[..., 2] = 1.0
            norm = quat_rotate(q, ref_norm)

            return torch.cat((tan, norm), dim=-1)

        joint_quat = quat_from_angle_axis(joint_positions.reshape(-1), self._joint_axes.repeat(joint_positions.shape[0], 1))
        joint_vec6 = quat_to_tan_norm(joint_quat)
        return joint_vec6.view(joint_positions.shape[0], -1)

    def sample_times(self, motion_ids: np.ndarray, duration: float | None = None) -> np.ndarray:
        """Sample random motion times."""
        clip_duration = self._motion_durations[motion_ids]
        if duration is not None:
            clip_duration = np.minimum(clip_duration, duration)
        return clip_duration * np.random.uniform(low=0.0, high=1.0, size=motion_ids.shape[0])

    def sample_motion_ids(self, num_samples: int) -> np.ndarray:
        """Sample motion ids using the per-motion JSON weights."""
        return np.random.choice(self._n_motions, size=num_samples, p=self._motion_probs)

    def sample(self, num_samples: int, motion_ids: np.ndarray | None = None, times: np.ndarray | None = None, duration: float | None = None, *,
        return_full_state: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample motion data at the requested times."""

        # 1. Select which motion clips each sample should come from.
        if motion_ids is None:
            motion_ids = self.sample_motion_ids(num_samples)
        else:
            motion_ids = np.asarray(motion_ids, dtype=np.int64)

        # 2. Select sampling times.
        sample_times = self.sample_times(motion_ids, duration) if times is None else np.asarray(times, dtype=np.float64)

        # 3. Convert times into frame indexes and blend coefficients.
        frame_start, frame_end, blend = self.compute_frame_blend(motion_ids, sample_times)
        blend = torch.tensor(blend, dtype=torch.float32, device=self._device)

        # 4. Gatehr start/end frames from the selected motion clips.
        start_frames = torch.tensor(
            [self._data[motion_id]["frames"][frame_idx] for motion_id, frame_idx in zip(motion_ids, frame_start, strict=False)],
            dtype=torch.float32,
            device=self._device,
        )
        end_frames = torch.tensor(
            [self._data[motion_id]["frames"][frame_idx] for motion_id, frame_idx in zip(motion_ids, frame_end, strict=False)],
            dtype=torch.float32,
            device=self._device,
        )

        def slice_field(frames: torch.Tensor, name: str) -> torch.Tensor:
            start = self._frame_offsets[name]
            size = self._frame_sizes[name]
            return frames[:, start : start + size]

        # 5. Interpolate each AMP observation component.
        root_position = self.interpolate(
            slice_field(start_frames, "root_position"),
            b=slice_field(end_frames, "root_position"),
            blend=blend,
        )
        root_orientation_quat = self.slerp(
            slice_field(start_frames, "root_orientation_quat"),
            q1=slice_field(end_frames, "root_orientation_quat"),
            blend=blend,
        )
        root_linear_velocity = self.interpolate(
            slice_field(start_frames, "world_linear_velocity"),
            b=slice_field(end_frames, "world_linear_velocity"),
            blend=blend,
        )
        local_root_angular_velocity = self.interpolate(
            slice_field(start_frames, "world_angular_velocity"),
            b=slice_field(end_frames, "world_angular_velocity"),
            blend=blend,
        )
        joints_positions = self.interpolate(
            slice_field(start_frames, "joints_positions"),
            b=slice_field(end_frames, "joints_positions"),
            blend=blend,
        )
        joints_positions_vectorized = self.vectorize_joint_positions(joints_positions)
        joints_velocities = self.interpolate(
            slice_field(start_frames, "joints_velocities"),
            b=slice_field(end_frames, "joints_velocities"),
            blend=blend,
        )
        left_toe_position = self.interpolate(
            slice_field(start_frames, "left_toe_position"),
            b=slice_field(end_frames, "left_toe_position"),
            blend=blend,
        )
        right_toe_position = self.interpolate(
            slice_field(start_frames, "right_toe_position"),
            b=slice_field(end_frames, "right_toe_position"),
            blend=blend,
        )

        if return_full_state:
            return (
                root_position,
                root_orientation_quat,
                root_linear_velocity,
                local_root_angular_velocity,
                joints_positions,
                joints_velocities,
                left_toe_position,
                right_toe_position,
            )

        (
            root_orientation_quat_normalized_heading,
            root_linear_velocity_normalized_heading,
            local_root_angular_velocity_normalized_heading,
        ) = self.normalize_heading_observation(
            root_orientation_quat,
            root_linear_velocity,
            local_root_angular_velocity,
        )
        root_height = root_position[:, 2:3]

        return (
            root_height,
            root_orientation_quat_normalized_heading,
            root_linear_velocity_normalized_heading,
            local_root_angular_velocity_normalized_heading,
            joints_positions_vectorized,
            joints_velocities,
            left_toe_position,
            right_toe_position,
        )
