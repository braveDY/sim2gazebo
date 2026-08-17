#!/usr/bin/env python3
import math
import os
import threading

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


def decode_layer(message, layer_name):
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
    raise ValueError("elevation layer has no 2D layout")


class ElevationMapAdapter(Node):
    def __init__(self):
        super().__init__("rl_elevation_map_adapter")
        self.declare_parameter("robot", "go2")
        self.declare_parameter("policy", "quad_mwm")
        self.declare_parameter("policy_root", "")
        self.declare_parameter("input_topic", "/filtered_map")
        self.declare_parameter("odom_topic", "/odom")
        root = self.get_parameter("policy_root").value
        if not root:
            root = os.path.join(get_package_share_directory("rl_policy_runtime"), "policy")
        policy_dir = os.path.join(root, self.get_parameter("robot").value,
                                  self.get_parameter("policy").value)
        with open(os.path.join(policy_dir, "manifest.yaml"), "r") as stream:
            manifest = yaml.safe_load(stream)
        config = manifest.get("sensors", {}).get("elevation_map", {})
        self.layer = config.get("layer", "elevation")
        self.resolution = float(config.get("resolution", 0.05))
        size_x, size_y = config.get("size", [1.6, 1.0])
        offset_x, offset_y = config.get("offset", [0.0, 0.0])
        x = np.arange(-size_x / 2, size_x / 2 + 1e-6, self.resolution) + offset_x
        y = np.arange(-size_y / 2, size_y / 2 + 1e-6, self.resolution) + offset_y
        gx, gy = np.meshgrid(x, y, indexing="xy")
        self.points = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)
        clip = config.get("clip", [-1.2, 0.0])
        self.clip = (float(clip[0]), float(clip[1]))
        self.output_topic = config.get("topic", "/ame_elevation_map")
        self._lock = threading.Lock()
        self._elevation = self._center = self._base = None
        self._resolution = self._yaw = None
        self.pub = self.create_publisher(Float32MultiArray, self.output_topic, 10)
        self.create_subscription(GridMap, self.get_parameter("input_topic").value, self.map_cb, 10)
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self.odom_cb, 20)
        self.create_timer(0.02, self.publish_map)
        self.get_logger().info("Elevation adapter: policy=%s, output=%s, points=%d" %
                               (self.get_parameter("policy").value, self.output_topic, len(self.points)))

    def map_cb(self, message):
        if self.layer not in message.layers:
            return
        try:
            elevation = decode_layer(message, self.layer)
        except (ValueError, IndexError) as exc:
            self.get_logger().warning("Ignoring elevation map: %s" % exc)
            return
        with self._lock:
            self._elevation = elevation
            self._center = np.array([message.info.pose.position.x, message.info.pose.position.y], dtype=np.float32)
            self._resolution = float(message.info.resolution)

    def odom_cb(self, message):
        q = message.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        with self._lock:
            self._base = np.array([message.pose.pose.position.x, message.pose.pose.position.y,
                                   message.pose.pose.position.z], dtype=np.float32)
            self._yaw = yaw

    def publish_map(self):
        with self._lock:
            if self._elevation is None or self._base is None or self._center is None:
                return
            elevation, center, resolution, base, yaw = self._elevation.copy(), self._center.copy(), self._resolution, self._base.copy(), self._yaw
        c, s = math.cos(yaw), math.sin(yaw)
        wx = base[0] + self.points[:, 0] * c - self.points[:, 1] * s
        wy = base[1] + self.points[:, 0] * s + self.points[:, 1] * c
        rows, cols = elevation.shape
        ri = np.rint(rows / 2 - 0.5 - (wx - center[0]) / resolution).astype(int)
        ci = np.rint(cols / 2 - 0.5 - (wy - center[1]) / resolution).astype(int)
        valid = (ri >= 0) & (ri < rows) & (ci >= 0) & (ci < cols)
        values = np.full(len(self.points), self.clip[0], dtype=np.float32)
        sampled = elevation[ri[valid], ci[valid]] - base[2]
        values[valid] = np.clip(np.nan_to_num(sampled, nan=self.clip[0]), *self.clip)
        xyz = np.column_stack((self.points, values)).astype(np.float32).reshape(-1)
        msg = Float32MultiArray()
        msg.layout.dim = [MultiArrayDimension(label="xyz", size=int(xyz.size), stride=int(xyz.size))]
        msg.data = xyz.tolist()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = ElevationMapAdapter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
