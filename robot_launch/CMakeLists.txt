cmake_minimum_required(VERSION 3.5)
project(robot_launch)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(ament_cmake REQUIRED)
find_package(robot_description REQUIRED)
find_package(robot_control REQUIRED)
find_package(commander REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

install(
  DIRECTORY
    launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install project configuration files
install(
  DIRECTORY
    worlds/
  DESTINATION share/${PROJECT_NAME}/worlds
)

ament_package()

