import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "robot",
            default_value=TextSubstitution(text="go2"),
            description="Robot name",
        ),
        DeclareLaunchArgument(
            "policy",
            default_value=TextSubstitution(text="isaaclab_ame"),
            description="Policy name to deploy",
        ),
        DeclareLaunchArgument(
            "policy_root",
            default_value=TextSubstitution(text=""),
            description="Root directory for policy packages",
        ),
        DeclareLaunchArgument(
            "control_enabled",
            default_value=TextSubstitution(text="false"),
            description="Enable control at startup",
        ),
        DeclareLaunchArgument(
            "keyboard_enabled",
            default_value=TextSubstitution(text="false"),
            description="Enable terminal keyboard input (use ros2 run for this)",
        ),
        Node(
            package="rl_policy_runtime",
            executable="deploy_node",
            name="rl_policy_runtime",
            output="screen",
            parameters=[{
                "robot": LaunchConfiguration("robot"),
                "policy": LaunchConfiguration("policy"),
                "policy_root": LaunchConfiguration("policy_root"),
                "control_enabled": LaunchConfiguration("control_enabled"),
                "keyboard_enabled": LaunchConfiguration("keyboard_enabled"),
                "command_topic": "/robot_joint_controller/command",
                "robot_state_topic": "/robot_joint_controller/state",
                "imu_topic": "/imu",
                "odom_topic": "/odom",
                "cmd_vel_topic": "/cmd_vel",
            }],
        ),
    ])
