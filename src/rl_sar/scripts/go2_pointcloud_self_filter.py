#!/usr/bin/env python3

import math
from typing import Iterable, List, Sequence, Tuple

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSPresetProfiles, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import tf2_ros


Point = Tuple[float, float, float]
Box = Tuple[float, float, float, float, float, float]


def rotate_point(point: Point, quaternion: Sequence[float]) -> Point:
    x, y, z = point
    qx, qy, qz, qw = quaternion
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    return (
        x + qw * tx + qy * tz - qz * ty,
        y + qw * ty + qz * tx - qx * tz,
        z + qw * tz + qx * ty - qy * tx,
    )


def transform_point(point: Point, transform: TransformStamped) -> Point:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    rotated = rotate_point(point, (rotation.x, rotation.y, rotation.z, rotation.w))
    return (
        rotated[0] + translation.x,
        rotated[1] + translation.y,
        rotated[2] + translation.z,
    )


def point_in_box(point: Point, box: Box) -> bool:
    x, y, z = point
    x_min, x_max, y_min, y_max, z_min, z_max = box
    return (
        x_min <= x <= x_max
        and y_min <= y <= y_max
        and z_min <= z <= z_max
    )


class Go2PointCloudSelfFilter(Node):
    def __init__(self) -> None:
        super().__init__("go2_pointcloud_self_filter")

        self.input_topic = self.declare_parameter("input_topic", "/utlidar/cloud").value
        self.output_topic = self.declare_parameter(
            "output_topic", "/utlidar/cloud_filtered"
        ).value
        self.base_frame = self.declare_parameter("base_frame", "base").value
        self.padding = float(self.declare_parameter("padding", 0.04).value)
        self.publish_removed = bool(
            self.declare_parameter("publish_removed", True).value
        )
        self.removed_topic = self.declare_parameter(
            "removed_topic", "/utlidar/cloud_self_points"
        ).value

        robot_box = (
            float(self.declare_parameter("robot_x_min", -0.45).value),
            float(self.declare_parameter("robot_x_max", 0.45).value),
            float(self.declare_parameter("robot_y_min", -0.30).value),
            float(self.declare_parameter("robot_y_max", 0.30).value),
            float(self.declare_parameter("robot_z_min", -0.70).value),
            float(self.declare_parameter("robot_z_max", 0.20).value),
        )
        self.robot_box = self.padded_box(robot_box)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        input_qos = QoSPresetProfiles.get_from_short_key("sensor_data")
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.publisher = self.create_publisher(PointCloud2, self.output_topic, output_qos)
        self.removed_publisher = None
        if self.publish_removed:
            self.removed_publisher = self.create_publisher(
                PointCloud2, self.removed_topic, output_qos
            )
        self.subscription = self.create_subscription(
            PointCloud2, self.input_topic, self.pointcloud_callback, input_qos
        )
        self.get_logger().info(
            f"Filtering {self.input_topic} -> {self.output_topic} in {self.base_frame}; "
            f"using one robot box "
            f"with {self.padding:.3f} m padding"
        )

    def padded_box(self, box: Box) -> Box:
        x_min, x_max, y_min, y_max, z_min, z_max = box
        return (
            x_min - self.padding,
            x_max + self.padding,
            y_min - self.padding,
            y_max + self.padding,
            z_min - self.padding,
            z_max + self.padding,
        )

    def point_is_self(self, point: Point) -> bool:
        return point_in_box(point, self.robot_box)

    def lookup_transform(self, message: PointCloud2) -> TransformStamped:
        if message.header.frame_id == self.base_frame:
            identity = TransformStamped()
            identity.transform.rotation.w = 1.0
            return identity
        return self.tf_buffer.lookup_transform(
            self.base_frame,
            message.header.frame_id,
            message.header.stamp,
        )

    def transform_points(
        self, points: Iterable[Point], transform: TransformStamped
    ) -> List[Point]:
        return [transform_point(point, transform) for point in points]

    def make_cloud(self, header, points: List[Point]) -> PointCloud2:
        return point_cloud2.create_cloud_xyz32(header, points)

    def pointcloud_callback(self, message: PointCloud2) -> None:
        try:
            transform = self.lookup_transform(message)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as error:
            self.get_logger().warn(f"Waiting for TF to {self.base_frame}: {error}")
            return

        raw_points: List[Point] = []
        for point in point_cloud2.read_points(
            message, field_names=("x", "y", "z"), skip_nans=True
        ):
            if getattr(point, "dtype", None) is not None and point.dtype.names:
                x_value = float(point["x"])
                y_value = float(point["y"])
                z_value = float(point["z"])
            else:
                x_value, y_value, z_value = (float(value) for value in point[:3])
            if all(math.isfinite(value) for value in (x_value, y_value, z_value)):
                raw_points.append((x_value, y_value, z_value))

        base_points = self.transform_points(raw_points, transform)
        filtered_points = []
        removed_points = []
        for raw_point, base_point in zip(raw_points, base_points):
            if self.point_is_self(base_point):
                removed_points.append(raw_point)
            else:
                filtered_points.append(raw_point)

        self.publisher.publish(self.make_cloud(message.header, filtered_points))
        if self.removed_publisher is not None:
            self.removed_publisher.publish(self.make_cloud(message.header, removed_points))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Go2PointCloudSelfFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
