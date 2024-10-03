import os
import sys
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.actions import RegisterEventHandler, SetEnvironmentVariable
from launch.actions import IncludeLaunchDescription, TimerAction, ExecuteProcess
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch.event_handlers import OnProcessExit

from launch_ros.actions import Node
import xacro

def generate_launch_description():
    
    pkg_robot_description = get_package_share_directory('robot_description')
    pkg_robot_launch = get_package_share_directory('robot_launch')
    pkg_commander = get_package_share_directory('commander')

    commander = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(pkg_commander, 'launch','commander.launch.py')]))

    resource_value = os.path.join(pkg_robot_launch, 'worlds') + ':' + str(Path(pkg_robot_description).parent.resolve())
    os.environ['GZ_SIM_RESOURCE_PATH'] = resource_value

    world = os.path.join(pkg_robot_launch,
        'worlds/poleTraining.sdf'
    )

    gz_sim_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch'), '/gz_sim.launch.py']),
        launch_arguments=[
            ('gz_args', [world,
                            ' -v 4',
                            ' -s',
                            ' -r']
            )
        ]
    )

    gz_sim_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch'), '/gz_sim.launch.py']),
        launch_arguments=[
            ('gz_args', [' -v 4',
                            ' -g']
            )
        ]
    )

    xacro_file = os.path.join(pkg_robot_description,
                              'robot',
                              'cart_pole.urdf.xacro')

    urdf_file_path = os.path.join(pkg_robot_description,
                                'robot',
                                'cart_pole.urdf')


    doc = xacro.process_file(xacro_file, mappings={'use_sim' : 'true'})

    robot_desc = doc.toprettyxml(indent='  ')

    try:
        # Save the URDF content to the output file
        with open(urdf_file_path, 'w') as urdf_file:
            urdf_file.write(robot_desc)        
        print(f"Successfully converted Xacro to URDF and saved to {urdf_file_path}")
    except Exception as e:
        print(f"Error converting Xacro to URDF: {e}", file=sys.stderr)


    params = {'robot_description': robot_desc}
    
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )


    gz_sim_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', '/robot_description',
                   '-x', '1.5',
                   '-y', '0.0',
                   '-z', '1.3610',
                   '-R', '0.0',
                   '-P', '0.0',
                   '-Y', '-1.0',
                   '-name', 'cart_pole',
                   '-allow_renaming', 'false'],
    )

     ## Ros2 Control Joint broad caster
    joint_position_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_position_controller"],
    )

    joint_velocity_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_velocity_controller"],
    )
    
    joint_broadcast_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
    )

    # Bridge
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
          arguments=[
            '/model/cart_pole/pose@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V',
            '/world/default/control@ros_gz_interfaces/srv/ControlWorld',
            '/world/default/create@ros_gz_interfaces/srv/SpawnEntity',
            '/world/default/create@ros_gz_interfaces/srv/SpawnEntity',
            '/world/default/remove@ros_gz_interfaces/srv/DeleteEntity',
        ],
        output='screen'
    )

 
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
        '/model/cart_pole/pose@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V',
        '/world/default/control@ros_gz_interfaces/srv/ControlWorld',
        '/world/default/create@ros_gz_interfaces/srv/SpawnEntity',
        '/world/default/remove@ros_gz_interfaces/srv/DeleteEntity',
        '/world/default/set_pose@ros_gz_interfaces/srv/SetEntityPose',
        ],
        output='screen'
    )

    return LaunchDescription([
         RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gz_sim_spawn_entity,
                on_exit=[joint_broadcast_controller_spawner],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
               target_action=joint_broadcast_controller_spawner,
               on_exit=[joint_position_controller_spawner],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
               target_action=joint_position_controller_spawner,
               on_exit=[commander],
            )
        ),
        node_robot_state_publisher,
        gz_sim_server,
        gz_sim_gui,
        gz_sim_spawn_entity,
        bridge
    ])
