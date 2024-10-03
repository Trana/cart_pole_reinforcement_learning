cmake_minimum_required(VERSION 3.5)
project(commander)


find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
# find_package(transforms3d REQUIRED)

install(
  DIRECTORY
    launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# ament_target_dependencies(${PROJECT_NAME}
#   tf_transformations)

install(
    PROGRAMS
    scripts/optimize_pid_learning.py
    scripts/Q_learning.py
    DESTINATION lib/${PROJECT_NAME}
  )

ament_package()