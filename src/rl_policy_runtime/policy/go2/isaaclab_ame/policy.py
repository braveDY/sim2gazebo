from typing import Any, Dict, Mapping

import numpy as np
import torch

from rl_policy_runtime.policy_api import Policy as BasePolicy, RobotState


class Policy(BasePolicy):
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        config: Dict[str, Any],
        gazebo_to_policy: np.ndarray,
        policy_to_gazebo: np.ndarray,
    ) -> None:
        self.model = model
        self.ang_vel_scale = float(config["ang_vel_scale"])
        self.dof_vel_scale = float(config["dof_vel_scale"])
        self.ame_default_dof_pos = np.array(config["ame_default_dof_pos"], dtype=np.float32)
        self.gazebo_to_policy = gazebo_to_policy
        self.policy_to_gazebo = policy_to_gazebo

    def reset(self) -> None:
        pass

    def _map_to_policy_order(self, values: np.ndarray) -> np.ndarray:
        return values[self.gazebo_to_policy]

    def _build_proprio(self, state: RobotState) -> np.ndarray:
        current_joint_pos = self._map_to_policy_order(state.joint_pos)
        current_joint_vel = self._map_to_policy_order(state.joint_vel)
        previous_action = self._map_to_policy_order(state.previous_action)

        angular_velocity = state.angular_velocity * self.ang_vel_scale

        projected_gravity = self._quat_rotate_inverse(
            state.quaternion_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32)
        )

        joint_pos_rel = current_joint_pos - self.ame_default_dof_pos
        joint_vel_scaled = current_joint_vel * self.dof_vel_scale

        proprio = np.concatenate([
            angular_velocity,
            projected_gravity,
            state.command,
            joint_pos_rel,
            joint_vel_scaled,
            previous_action,
        ]).astype(np.float32)
        return proprio

    def infer(
        self,
        state: RobotState,
        sensors: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        elevation_map = sensors["elevation_map"].astype(np.float32).reshape(-1)

        proprio = self._build_proprio(state)

        observation = np.concatenate([proprio, elevation_map], dtype=np.float32)

        with torch.no_grad():
            action = self.model(
                torch.from_numpy(observation).unsqueeze(0)
            )

        return action.squeeze(0).cpu().numpy()

    @staticmethod
    def _quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]
        q_conj = np.array([q_w, -q_x, -q_y, -q_z], dtype=np.float32)
        return Policy._quat_rotate(q_conj, v)

    @staticmethod
    def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]
        v_x, v_y, v_z = v[0], v[1], v[2]

        t_x = 2.0 * (q_y * v_z - q_z * v_y)
        t_y = 2.0 * (q_z * v_x - q_x * v_z)
        t_z = 2.0 * (q_x * v_y - q_y * v_x)

        return np.array([
            v_x + q_w * t_x + (q_y * t_z - q_z * t_y),
            v_y + q_w * t_y + (q_z * t_x - q_x * t_z),
            v_z + q_w * t_z + (q_x * t_y - q_y * t_x),
        ], dtype=np.float32)
