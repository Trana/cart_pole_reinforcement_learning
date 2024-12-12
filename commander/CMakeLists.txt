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
    scripts/helpers/timing_test.py
    scripts/pid_learning.py
    scripts/q_learning.py
    scripts/dqn_learning.py
    scripts/ddqn_learning.py
    scripts/common/pose_subscriber.py
    scripts/common/replay_buffer.py
    scripts/common/shared_state.py
    scripts/common/training_config.py
    scripts/common/fcq_network.py
    scripts/common/strategies/e_greedy_exp_strategy.py
    scripts/common/strategies/e_greedy_linear_strategy.py
    scripts/common/strategies/e_greedy_strategy.py
    scripts/common/strategies/greedy_strategy.py
    DESTINATION lib/${PROJECT_NAME}
  )

ament_package()