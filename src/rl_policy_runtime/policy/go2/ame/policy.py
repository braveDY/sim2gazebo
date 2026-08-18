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
        self.dof_vel_scale = float(config.get("dof_vel_scale", 0.05))
        self.ame_default_dof_pos = np.array(config["ame_default_dof_pos"], dtype=np.float32)
        self.gazebo_to_policy = gazebo_to_policy
        self.policy_to_gazebo = policy_to_gazebo

        size = config.get("elevation_size", [1.2, 0.8])
        offset = config.get("elevation_offset", [0.375, 0.0])
        resolution = float(config.get("elevation_resolution", 0.05))
        self.points_xy = create_grid_points(size=size, resolution=resolution, offset=offset)
        self.clip = tuple(float(v) for v in config.get("elevation_clip", [-1.2, 0.0]))
        self.layer = str(config.get("elevation_layer", "elevation"))

    def infer(
        self,
        state: RobotState,
        sensors: Mapping[str, Any],
    ) -> np.ndarray:
        current_joint_pos = state.joint_pos[self.gazebo_to_policy]
        current_joint_vel = state.joint_vel[self.gazebo_to_policy]
        previous_action = state.previous_action[self.gazebo_to_policy]
        proprio = np.concatenate([
            state.angular_velocity * self.ang_vel_scale,
            project_gravity(state.quaternion_wxyz),
            state.command,
            current_joint_pos - self.ame_default_dof_pos,
            current_joint_vel * self.dof_vel_scale,
            previous_action,
        ]).astype(np.float32)

        grid_map_msg = sensors.get("elevation_map")
        if grid_map_msg is not None:
            yaw = quat_to_euler_xyz(state.quaternion_wxyz)[2]
            sampled_z = sample_grid_map(
                grid_map_msg, state.base_position, yaw, self.points_xy, self.layer, self.clip
            )
            elevation_map = np.column_stack((self.points_xy, sampled_z)).astype(np.float32).reshape(-1)
        else:
            elevation_map = np.zeros(len(self.points_xy) * 3, dtype=np.float32)

        observation = np.concatenate([proprio, elevation_map], dtype=np.float32)

        with torch.no_grad():
            action = self.model(torch.from_numpy(observation).unsqueeze(0))

        return action.squeeze(0).cpu().numpy().astype(np.float32)
