from typing import Any, Dict, Mapping
from collections import deque
import numpy as np
import torch

from rl_policy_runtime.policy_api import (
    Policy as BasePolicy,
    RobotState,
    create_grid_points,
    quat_to_euler_xyz,
    sample_grid_map,
)


class HistoryBuffer:
    def __init__(self, history_len: int = 1) -> None:
        self.history_len = max(1, int(history_len))
        self._buffer: deque = deque(maxlen=self.history_len)

    def reset(self) -> None:
        self._buffer.clear()

    @property
    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    def append(self, item: np.ndarray) -> None:
        if not self._buffer:
            for _ in range(self.history_len):
                self._buffer.append(item.copy().astype(np.float32))
        else:
            self._buffer.append(item.copy().astype(np.float32))

    def get_flattened(self) -> np.ndarray:
        if not self._buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(list(self._buffer), dtype=np.float32)


class Policy(BasePolicy):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        config: Dict[str, Any],
        gazebo_to_policy: np.ndarray,
        policy_to_gazebo: np.ndarray,
    ) -> None:
        super().__init__(model, config, gazebo_to_policy, policy_to_gazebo)
        self.ang_vel_scale = float(config.get("ang_vel_scale", 1.0))
        self.dof_vel_scale = float(config.get("dof_vel_scale", 0.05))
        self.ame_default_dof_pos = np.array(config["ame_default_dof_pos"], dtype=np.float32)
        self.history_buffer = HistoryBuffer(history_len=int(config.get("proprio_history_length", 4)))

        size = config.get("elevation_size", [1.2, 0.8])
        offset = config.get("elevation_offset", [0.375, 0.0])
        resolution = float(config.get("elevation_resolution", 0.05))
        self.points_xy = create_grid_points(size=size, resolution=resolution, offset=offset)
        self.clip = tuple(float(v) for v in config.get("elevation_clip", [-1.2, 0.0]))
        self.layer = str(config.get("elevation_layer", "elevation"))

    def reset(self) -> None:
        self.history_buffer.reset()

    def infer(
        self,
        state: RobotState,
        sensors: Mapping[str, Any],
    ) -> np.ndarray:
        proprio = self.build_proprio_features(
            state=state,
            default_joint_pos=self.ame_default_dof_pos,
            ang_vel_scale=self.ang_vel_scale,
            dof_vel_scale=self.dof_vel_scale,
            include_lin_vel=False,
        )

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

        if self.history_buffer.is_empty:
            self.history_buffer.append(proprio)
        history = self.history_buffer.get_flattened()
        with torch.no_grad():
            action = self.model(
                torch.from_numpy(observation).unsqueeze(0),
                torch.from_numpy(history).unsqueeze(0),
            )
        self.history_buffer.append(proprio)

        return action.squeeze(0).cpu().numpy().astype(np.float32)