cmake_minimum_required(VERSION 3.5)
project(robot_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY
    urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
)

install(
  DIRECTORY
    robot/
  DESTINATION share/${PROJECT_NAME}/robot
)

install(
  DIRECTORY
    meshes/
  DESTINATION share/${PROJECT_NAME}/meshes
)

ament_package()
