import rclpy
from ros_gz_interfaces.srv import ControlWorld, DeleteEntity, SpawnEntity, SetEntityPose
from geometry_msgs.msg import Pose, Point, Quaternion
from ros_gz_interfaces.msg import Entity

class GazeboServiceHelper:
    def __init__(self, node):
        self.node = node

        # Initialize service clients
        self.reset_simulation_client = self.node.create_client(ControlWorld, '/world/default/control')
        self.remove_client = self.node.create_client(DeleteEntity, '/world/default/remove')
        self.create_entity_client = self.node.create_client(SpawnEntity, '/world/default/create')
        self.set_pose_client = self.node.create_client(SetEntityPose, '/world/default/set_pose')

        # Wait for services to be available
        self.wait_for_service(self.remove_client, 'remove')
        self.wait_for_service(self.create_entity_client, 'create')
        self.wait_for_service(self.set_pose_client, 'set_pose')

    def wait_for_service(self, client, service_name):
        while not client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info(f'Service /world/default/{service_name} not available, waiting again...')

    def set_pose(self, entity_name, position=None, orientation=None):
        """Set the pose of an entity in the Gazebo world."""
        position = position or [0.2, 0, 0]
        orientation = orientation or [0.0, 0.0, 0.0, 0.0]

        request = SetEntityPose.Request()
        request.entity = Entity()
        request.entity.name = entity_name
        request.pose = Pose()
        request.pose.position = Point(x=position[0], y=position[1], z=position[2])
        request.pose.orientation = Quaternion(x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3])

        future = self.set_pose_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        response = future.result()

        if response and response.success:
            self.node.get_logger().info(f'Successfully set pose for {entity_name}')
        else:
            self.node.get_logger().error(f'Failed to set pose for {entity_name}')

    def delete_entity(self, entity_name):
        """Delete an entity from the Gazebo world."""
        request = DeleteEntity.Request()
        request.entity.name = entity_name

        future = self.remove_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        response = future.result()

        if response and response.success:
            self.node.get_logger().info(f'Successfully removed {entity_name}')
        else:
            self.node.get_logger().error(f'Failed to remove {entity_name}')

    def create_entity(self, entity_name, sdf_file_path, position=[0, 0, 0], orientation=[0, 0, 0, 1]):
        """Spawn an entity in the Gazebo world."""
        request = SpawnEntity.Request()
        request.entity_factory.name = entity_name
        request.entity_factory.sdf_filename = sdf_file_path
        request.entity_factory.pose.position = Point(x=position[0], y=position[1], z=position[2])
        request.entity_factory.pose.orientation = Quaternion(x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3])

        future = self.create_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        response = future.result()

        if response and response.success:
            self.node.get_logger().info(f'Successfully spawned {entity_name}')
        else:
            self.node.get_logger().error(f'Failed to spawn {entity_name}')

    def pause_simulation(self):
        """Pause the Gazebo simulation."""
        request = ControlWorld.Request()
        request.world_control.pause = True

        future = self.reset_simulation_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        response = future.result()

        if response and response.success:
            self.node.get_logger().info('Simulation paused successfully.')
        else:
            self.node.get_logger().error('Failed to pause the simulation.')

    def unpause_simulation(self):
        """Unpause the Gazebo simulation."""
        request = ControlWorld.Request()
        request.world_control.pause = False

        future = self.reset_simulation_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        response = future.result()

        if response and response.success:
            self.node.get_logger().info('Simulation unpaused successfully.')
        else:
            self.node.get_logger().error('Failed to unpause the simulation.')

    def reset_simulation(self, reset_time_only=True, reset_model_only=False):
        """Reset the Gazebo simulation."""
        request = ControlWorld.Request()
        request.world_control.reset.time_only = reset_time_only
        request.world_control.reset.all = not reset_model_only
        request.world_control.reset.model_only = reset_model_only

        future = self.reset_simulation_client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        response = future.result()

        if response and response.success:
            self.node.get_logger().info('Simulation reset successfully.')
        else:
            self.node.get_logger().error('Failed to reset the simulation.')


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('gazebo_service_helper_node')

    # Create the helper with the existing node
    helper = GazeboServiceHelper(node)

    # Example usage of helper class
    helper.pause_simulation()
    helper.create_entity("cart_pole", "/path/to/cart_pole.urdf")
    helper.set_pose("cart_pole", position=[1, 0, 2], orientation=[0, 0, 0, 1])
    helper.unpause_simulation()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()