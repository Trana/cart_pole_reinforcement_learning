import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument, TimerAction
from launch.actions import RegisterEventHandler, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch.event_handlers import OnProcessExit

from launch_ros.actions import Node
import xacro

def generate_launch_description():
# pose_subscriber
  return LaunchDescription([
          Node(package='commander', executable='optimize_pid_learning.py', output='screen'),         
      ])