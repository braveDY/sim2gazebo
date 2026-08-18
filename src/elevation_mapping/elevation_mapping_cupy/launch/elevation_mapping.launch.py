import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression


def generate_launch_description():
    package_name = 'elevation_mapping_cupy'
    share_dir = get_package_share_directory(package_name)

    # Define paths
    core_param_path = os.path.join(
        share_dir, 'config', 'core', 'core_param.yaml')

    # Declare launch arguments
    no_odom_arg = DeclareLaunchArgument(
        'no_odom',
        default_value='true',
        description='Whether to run in no-odom mode using base frame'
    )

    robot_param_arg = DeclareLaunchArgument(
        'robot_config',
        default_value=PythonExpression([
            "'go2/go2_gazebo_l1_no_odom.yaml' if '",
            LaunchConfiguration('no_odom'),
            "'.lower() in ['true', '1'] else 'go2/go2_gazebo_l1.yaml'"
        ]),
        description='Name of the robot-specific config file within '
                    'config/setups/'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock if true'
    )

    # Get launch configurations
    robot_config = LaunchConfiguration('robot_config')
    robot_param_path = PathJoinSubstitution(
        [share_dir, 'config', 'setups', robot_config])
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Verify core config exists
    if not os.path.exists(core_param_path):
        raise FileNotFoundError(
            f"Config file {core_param_path} does not exist")

    # Define nodes
    elevation_mapping_node = Node(
        package=package_name,
        executable='elevation_mapping_node.py',
        name='elevation_mapping_node',
        output='screen',
        parameters=[
            core_param_path,
            robot_param_path,
            {'use_sim_time': use_sim_time}
        ]
    )

    return LaunchDescription([
        no_odom_arg,
        robot_param_arg,
        use_sim_time_arg,
        elevation_mapping_node,
    ])
