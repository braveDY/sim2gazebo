cmake_minimum_required(VERSION 3.10)
project(rl_sar)

find_package(ament_cmake REQUIRED)

install(
  PROGRAMS
    scripts/gazebo_truth_tf.py
    scripts/go2_pointcloud_self_filter.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch rviz worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
