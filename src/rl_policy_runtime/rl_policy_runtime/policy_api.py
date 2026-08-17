from dataclasses import dataclass, field
from typing import Any, Dict, Mapping
import numpy as np

@dataclass
class RobotState:
    joint_pos: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32))
    joint_vel: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32))
    quaternion_wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    command: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    previous_action: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32))

def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
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


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)
    return quat_rotate(q_conj, v)


def project_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse(quaternion_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32))


def quat_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[0], q[1], q[2], q[3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float32)



class Policy:
    def __init__(
        self,
        model: Any,
        config: Dict[str, Any],
        gazebo_to_policy: np.ndarray,
        policy_to_gazebo: np.ndarray,
    ) -> None:
        self.model = model
        self.config = config
        self.gazebo_to_policy = gazebo_to_policy
        self.policy_to_gazebo = policy_to_gazebo

    def reset(self) -> None:
        pass

    def map_to_policy(self, values: np.ndarray) -> np.ndarray:
        return values[self.gazebo_to_policy]

    def map_to_gazebo(self, values: np.ndarray) -> np.ndarray:
        return values[self.policy_to_gazebo]

    def build_proprio_features(
        self,
        state: RobotState,
        default_joint_pos: np.ndarray,
        ang_vel_scale: float = 1.0,
        dof_vel_scale: float = 1.0,
        include_lin_vel: bool = False,
    ) -> np.ndarray:
        cur_pos = self.map_to_policy(state.joint_pos)
        cur_vel = self.map_to_policy(state.joint_vel)
        prev_act = self.map_to_policy(state.previous_action)

        components = []
        if include_lin_vel:
            components.append(state.linear_velocity)

        components.extend([
            state.angular_velocity * ang_vel_scale,
            project_gravity(state.quaternion_wxyz),
            state.command,
            cur_pos - default_joint_pos,
            cur_vel * dof_vel_scale,
            prev_act,
        ])
        return np.concatenate(components, dtype=np.float32)

    def infer(
        self,
        state: RobotState,
        sensors: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        raise NotImplementedError
