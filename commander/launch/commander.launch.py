import os
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
# pose_subscriber
  return LaunchDescription([
          Node(package='commander', executable='ddqn_learning.py', output='screen'),         
      ])