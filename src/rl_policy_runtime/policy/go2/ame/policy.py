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
        self.ang_vel_scale = float(config.get("ang_vel_scale", 1.0))
        self.dof_vel_scale = float(config.get("dof_vel_scale", 0.05))
        self.ame_default_dof_pos = np.array(config["ame_default_dof_pos"], dtype=np.float32)
        self.gazebo_to_policy = gazebo_to_policy
        self.policy_to_gazebo = policy_to_gazebo

    def infer(
        self,
        state: RobotState,
        sensors: Mapping[str, np.ndarray],
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
        elevation_map = sensors["elevation_map"].astype(np.float32).reshape(-1)
        observation = np.concatenate([proprio, elevation_map], dtype=np.float32)

        with torch.no_grad():
            action = self.model(torch.from_numpy(observation).unsqueeze(0))

        return action.squeeze(0).cpu().numpy().astype(np.float32)
