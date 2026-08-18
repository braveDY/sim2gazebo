from typing import Any, Dict, Mapping

import numpy as np
import torch

from rl_policy_runtime.policy_api import (
    Policy as BasePolicy,
    RobotState,
    create_grid_points,
    project_gravity,
    quat_to_euler_xyz,
    sample_grid_map,
)


class Policy(BasePolicy):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        config: Dict[str, Any],
        gazebo_to_policy: np.ndarray,
        policy_to_gazebo: np.ndarray,
    ) -> None:
        self.model = model
        self.default_joint_pos = np.asarray(config["default_joint_pos"], dtype=np.float32)
        self.height_scan_offset = float(config.get("height_scan_offset", 0.5))
        self.height_scan_shape = tuple(int(value) for value in config.get("height_scan_shape", [17, 11]))
        self.gazebo_to_policy = gazebo_to_policy
        self.policy_to_gazebo = policy_to_gazebo

        size = config.get("elevation_size", [1.6, 1.0])
        offset = config.get("elevation_offset", [0.0, 0.0])
        resolution = float(config.get("elevation_resolution", 0.1))
        self.points_xy = create_grid_points(size=size, resolution=resolution, offset=offset)
        self.clip = tuple(float(v) for v in config.get("elevation_clip", [-1.2, 0.0]))
        self.layer = str(config.get("elevation_layer", "elevation"))

    def _height_scan(self, z_rel: np.ndarray) -> np.ndarray:
        expected_size = int(np.prod(self.height_scan_shape))
        if z_rel.size != expected_size:
            raise ValueError(f"Expected {expected_size} height samples, got {z_rel.size}")
        return np.clip(-z_rel - self.height_scan_offset, -1.0, 1.0)

    def infer(self, state: RobotState, sensors: Mapping[str, Any]) -> np.ndarray:
        current_joint_pos = state.joint_pos[self.gazebo_to_policy]
        current_joint_vel = state.joint_vel[self.gazebo_to_policy]
        previous_action = state.previous_action[self.gazebo_to_policy]

        grid_map_msg = sensors.get("elevation_map")
        if grid_map_msg is not None:
            yaw = quat_to_euler_xyz(state.quaternion_wxyz)[2]
            z_rel = sample_grid_map(
                grid_map_msg, state.base_position, yaw, self.points_xy, self.layer, self.clip
            )
        else:
            z_rel = np.zeros(len(self.points_xy), dtype=np.float32)

        height_scan = self._height_scan(z_rel)

        observation = np.concatenate([
            state.linear_velocity,
            state.angular_velocity,
            project_gravity(state.quaternion_wxyz),
            state.command,
            current_joint_pos - self.default_joint_pos,
            current_joint_vel,
            previous_action,
            height_scan,
        ], dtype=np.float32)

        if observation.size != 235:
            raise ValueError(f"Expected 235 policy observations, got {observation.size}")
        with torch.no_grad():
            action = self.model(torch.from_numpy(observation).unsqueeze(0))
        return action.squeeze(0).cpu().numpy().astype(np.float32)
