cmake_minimum_required(VERSION 3.14)
project(cgal5_catkin)

find_package(ament_cmake REQUIRED)
find_package(CGAL 5 REQUIRED)

ament_export_dependencies(CGAL)
ament_package()
