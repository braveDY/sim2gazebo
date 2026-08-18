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
    base_position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

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


def decode_grid_map_layer(message: Any, layer_name: str) -> np.ndarray:
    """Decode a 2D layer numpy array from a grid_map_msgs/msg/GridMap message."""
    if layer_name not in message.layers:
        raise KeyError(f"Layer '{layer_name}' not found in GridMap layers: {list(message.layers)}")
    index = list(message.layers).index(layer_name)
    layer = message.data[index]
    values = np.asarray(layer.data, dtype=np.float32)
    dims = layer.layout.dim
    if len(dims) >= 2 and dims[0].label == "column_index":
        cols, rows = dims[0].size, dims[1].size
        return values.reshape((rows, cols), order="F")
    if len(dims) >= 2:
        rows, cols = dims[0].size, dims[1].size
        return values.reshape((rows, cols), order="C")
    raise ValueError(f"Layer '{layer_name}' has no 2D layout in GridMap message")


def sample_grid_map(
    grid_map_msg: Any,
    base_position: np.ndarray,
    yaw: float,
    points_xy: np.ndarray,
    layer_name: str = "elevation",
    clip: tuple = (-1.2, 0.0),
) -> np.ndarray:
    """Sample relative height (elevation - base_z) at specified (x,y) points in robot base frame.

    Args:
        grid_map_msg: grid_map_msgs.msg.GridMap message
        base_position: (3,) array with robot [x, y, z] in world/odom frame
        yaw: robot yaw angle in radians
        points_xy: (N, 2) array of query points in robot base frame
        layer_name: layer name to sample from
        clip: (min_h, max_h) height clip range

    Returns:
        (N,) array of relative height values in base frame, clipped to clip range.
    """
    elevation = decode_grid_map_layer(grid_map_msg, layer_name)
    center = np.array(
        [grid_map_msg.info.pose.position.x, grid_map_msg.info.pose.position.y],
        dtype=np.float32,
    )
    resolution = float(grid_map_msg.info.resolution)

    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    wx = base_position[0] + points_xy[:, 0] * c - points_xy[:, 1] * s
    wy = base_position[1] + points_xy[:, 0] * s + points_xy[:, 1] * c

    rows, cols = elevation.shape
    ri = np.rint(rows / 2 - 0.5 - (wx - center[0]) / resolution).astype(int)
    ci = np.rint(cols / 2 - 0.5 - (wy - center[1]) / resolution).astype(int)

    valid = (ri >= 0) & (ri < rows) & (ci >= 0) & (ci < cols)
    values = np.full(len(points_xy), clip[0], dtype=np.float32)
    sampled = elevation[ri[valid], ci[valid]] - base_position[2]
    values[valid] = np.clip(np.nan_to_num(sampled, nan=clip[0]), *clip)
    return values


def create_grid_points(
    size: tuple = (1.6, 1.0),
    resolution: float = 0.1,
    offset: tuple = (0.0, 0.0),
) -> np.ndarray:
    """Create a 2D grid point array (N, 2) centered at offset with given size and resolution."""
    size_x, size_y = float(size[0]), float(size[1])
    offset_x, offset_y = float(offset[0]), float(offset[1])
    x = np.arange(-size_x / 2, size_x / 2 + 1e-6, resolution) + offset_x
    y = np.arange(-size_y / 2, size_y / 2 + 1e-6, resolution) + offset_y
    gx, gy = np.meshgrid(x, y, indexing="xy")
    return np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1).astype(np.float32)


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
