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
        self.ang_vel_scale = float(config.get("ang_vel_scale", 1.0))
        self.dof_vel_scale = float(config.get("dof_vel_scale", 1.0))
        self.height_offset = float(config.get("height_offset", 0.5))
        self.grid_shape = tuple(int(x) for x in config.get("grid_shape", [17, 25]))
        self.cnn_default_dof_pos = np.array(config["cnn_default_dof_pos"], dtype=np.float32)
        self.gazebo_to_policy = gazebo_to_policy
        self.policy_to_gazebo = policy_to_gazebo

        size = config.get("elevation_size", [1.2, 0.8])
        offset = config.get("elevation_offset", [0.375, 0.0])
        resolution = float(config.get("elevation_resolution", 0.05))
        self.points_xy = create_grid_points(size=size, resolution=resolution, offset=offset)
        self.clip = tuple(float(v) for v in config.get("elevation_clip", [-1.2, 0.0]))
        self.layer = str(config.get("elevation_layer", "elevation"))

    def _build_obs_2d(self, z_rel: np.ndarray) -> np.ndarray:
        height_values = np.clip(-z_rel - self.height_offset, -1.0, 1.0).astype(np.float32)
        return height_values.reshape(1, 1, *self.grid_shape)

    def infer(
        self,
        state: RobotState,
        sensors: Mapping[str, Any],
    ) -> np.ndarray:
        current_joint_pos = state.joint_pos[self.gazebo_to_policy]
        current_joint_vel = state.joint_vel[self.gazebo_to_policy]
        previous_action = state.previous_action[self.gazebo_to_policy]
        obs_1d = np.concatenate([
            state.linear_velocity,
            state.angular_velocity * self.ang_vel_scale,
            project_gravity(state.quaternion_wxyz),
            state.command,
            current_joint_pos - self.cnn_default_dof_pos,
            current_joint_vel * self.dof_vel_scale,
            previous_action,
        ]).astype(np.float32)

        grid_map_msg = sensors.get("elevation_map")
        if grid_map_msg is not None:
            yaw = quat_to_euler_xyz(state.quaternion_wxyz)[2]
            z_rel = sample_grid_map(
                grid_map_msg, state.base_position, yaw, self.points_xy, self.layer, self.clip
            )
        else:
            z_rel = np.zeros(len(self.points_xy), dtype=np.float32)

        obs_2d = self._build_obs_2d(z_rel)

        obs_1d_t = torch.from_numpy(obs_1d).unsqueeze(0)
        obs_2d_t = [torch.from_numpy(obs_2d)]

        with torch.no_grad():
            action = self.model(obs_1d_t, obs_2d_t)

        return action.squeeze(0).cpu().numpy().astype(np.float32)
