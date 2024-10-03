cmake_minimum_required(VERSION 3.5)
project(robot_control)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(ament_cmake)

install(
  DIRECTORY
    config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()
