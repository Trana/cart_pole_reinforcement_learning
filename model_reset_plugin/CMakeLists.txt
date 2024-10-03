cmake_minimum_required(VERSION 3.10)

# Define the project name
project(model_reset_plugin)

# Find required packages
find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_srvs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Set Gazebo version variables
set(GZ_SIM_VER ${gz-sim8_VERSION_MAJOR})
set(GZ_PLUGIN_VER ${gz-plugin2_VERSION_MAJOR})

# Build the plugin library
add_library(${PROJECT_NAME} SHARED src/ModelResetPlugin.cc)

# Set C++ standard
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

# Apply dependencies to the target library
ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  std_srvs
  std_msgs
  geometry_msgs)

# Link against Gazebo libraries
target_link_libraries(${PROJECT_NAME}
  gz-sim${GZ_SIM_VER}::gz-sim${GZ_SIM_VER}
  gz-plugin${GZ_PLUGIN_VER}::gz-plugin${GZ_PLUGIN_VER})

# Install the plugin library
install(TARGETS ${PROJECT_NAME} DESTINATION lib/${PROJECT_NAME})