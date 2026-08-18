# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution, Command, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    rname = LaunchConfiguration("rname")
    wname = LaunchConfiguration("wname")
    enable_truth_tf = LaunchConfiguration("enable_truth_tf")

    rviz_config = PathJoinSubstitution([
        FindPackageShare("rl_sar"),
        "rviz",
        PythonExpression([
            "'go2.rviz' if '", enable_truth_tf, "'.lower() in ['true', '1'] else 'go2_no_odom.rviz'"
        ]),
    ])
    robot_name = ParameterValue(Command(["echo -n ", rname]), value_type=str)
    gazebo_model_name = ParameterValue(Command(["echo -n ", rname, "_gazebo"]), value_type=str)

    robot_description = ParameterValue(
        Command([
            "xacro ",
            Command(["echo -n ", Command(["ros2 pkg prefix ", rname, "_description"])]),
            "/share/", rname, "_description/xacro/robot.xacro"
        ]),
        value_type=str
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    world_path = PathJoinSubstitution([
        FindPackageShare("rl_sar"),
        "worlds",
        PythonExpression(["'", wname, ".world'"]),
    ])

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("gazebo_ros"), "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            # "verbose": "true",
            # "pause": "true",  # Not Available
            "world": world_path,
            "gui": "true",
        }.items(),
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", "/robot_description",
            "-entity", "robot_model",
            "-z", "0.4",
        ],
        output="screen",
    )

    lidar_frame_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="utlidar_lidar_frame",
        arguments=[
            "--frame-id", "radar",
            "--child-frame-id", "utlidar_lidar",
        ],
        output="screen",
    )

    pointcloud_self_filter_node = Node(
        package="rl_sar",
        executable="go2_pointcloud_self_filter.py",
        name="go2_pointcloud_self_filter",
        output="screen",
        parameters=[{
            "input_topic": "/utlidar/cloud",
            "output_topic": "/utlidar/cloud_filtered",
            "removed_topic": "/utlidar/cloud_self_points",
            "base_frame": "base",
            "padding": 0.04,
            "use_sim_time": True,
        }],
    )

    joint_state_broadcaster_node = Node(
        package="controller_manager",
        executable='spawner',
        arguments=["joint_state_broadcaster"],
        output="screen",
    )

    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        parameters=[{
            'deadzone': 0.1,
            'autorepeat_rate': 0.0,
        }],
    )

    ground_truth_node = Node(
        package="rl_sar",
        executable="gazebo_truth_tf.py",
        name="gazebo_truth_tf",
        output="screen",
        parameters=[{
            "odom_frame": "odom",
            "base_frame": "base",
            "use_sim_time": True,
        }],
        condition=IfCondition(enable_truth_tf),
    )

    param_node = Node(
        package="demo_nodes_cpp",
        executable="parameter_blackboard",
        name="param_node",
        parameters=[{
            "robot_name": robot_name,
            "gazebo_model_name": gazebo_model_name,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "rname",
            description="Robot name (go2)",
            default_value=TextSubstitution(text="go2"),
        ),
        DeclareLaunchArgument(
            "wname",
            description="Gazebo world name (stairs, terrain_track)",
            default_value=TextSubstitution(text="terrain_track"),
        ),
        DeclareLaunchArgument(
            "enable_truth_tf",
            description="Enable Gazebo ground truth TF broadcast and /odom publisher",
            default_value=TextSubstitution(text="false"),
        ),
        robot_state_publisher_node,
        gazebo,
        rviz_node,
        spawn_entity,
        lidar_frame_node,
        pointcloud_self_filter_node,
        joint_state_broadcaster_node,
        joy_node,
        ground_truth_node,
        param_node,
    ])
