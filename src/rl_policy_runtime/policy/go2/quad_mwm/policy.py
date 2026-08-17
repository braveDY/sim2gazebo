from typing import Any, Dict, Mapping
from collections import deque
import numpy as np
import torch

from rl_policy_runtime.policy_api import Policy as BasePolicy, RobotState

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

    def reset(self) -> None:
        self.history_buffer.reset()

    def infer(
        self,
        state: RobotState,
        sensors: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        proprio = self.build_proprio_features(
            state=state,
            default_joint_pos=self.ame_default_dof_pos,
            ang_vel_scale=self.ang_vel_scale,
            dof_vel_scale=self.dof_vel_scale,
            include_lin_vel=False,
        )
        elevation_map = sensors["elevation_map"].astype(np.float32).reshape(-1)
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