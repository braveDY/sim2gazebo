from typing import Any, Dict, Mapping

import numpy as np
import torch

from rl_policy_runtime.policy_api import Policy as BasePolicy, RobotState, project_gravity


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

    def _height_scan(self, sensor: np.ndarray) -> np.ndarray:
        values = np.asarray(sensor, dtype=np.float32).reshape(-1, 3)[:, 2]
        expected_size = int(np.prod(self.height_scan_shape))
        if values.size != expected_size:
            raise ValueError(f"Expected {expected_size} height samples, got {values.size}")
        return np.clip(-values - self.height_scan_offset, -1.0, 1.0)

    def infer(self, state: RobotState, sensors: Mapping[str, np.ndarray]) -> np.ndarray:
        current_joint_pos = state.joint_pos[self.gazebo_to_policy]
        current_joint_vel = state.joint_vel[self.gazebo_to_policy]
        previous_action = state.previous_action[self.gazebo_to_policy]
        observation = np.concatenate([
            state.linear_velocity,
            state.angular_velocity,
            project_gravity(state.quaternion_wxyz),
            state.command,
            current_joint_pos - self.default_joint_pos,
            current_joint_vel,
            previous_action,
            self._height_scan(sensors["elevation_map"]),
        ], dtype=np.float32)

        if observation.size != 235:
            raise ValueError(f"Expected 235 policy observations, got {observation.size}")
        with torch.no_grad():
            action = self.model(torch.from_numpy(observation).unsqueeze(0))
        return action.squeeze(0).cpu().numpy().astype(np.float32)
