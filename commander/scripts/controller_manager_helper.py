import rclpy
from rclpy.node import Node
from controller_manager_msgs.srv import LoadController, ConfigureController, SwitchController, ListControllers

class ControllerManagerHelper:
    def __init__(self, node: Node):
        self.node = node

        # Initialize the service clients for interacting with the controller manager
        self.load_controller_client = self.node.create_client(LoadController, '/controller_manager/load_controller')
        self.configure_controller_client = self.node.create_client(ConfigureController, '/controller_manager/configure_controller')
        self.switch_controller_client = self.node.create_client(SwitchController, '/controller_manager/switch_controller')
        self.list_controllers_client = self.node.create_client(ListControllers, '/controller_manager/list_controllers')

    def load_and_configure_controller(self, controller_name: str, should_start_controller: bool):
        # Load the controller
        self.node.get_logger().info(f"Trying to load controller {controller_name}")
        load_request = LoadController.Request()
        load_request.name = controller_name

        load_future = self.load_controller_client.call_async(load_request)
        rclpy.spin_until_future_complete(self.node, load_future)

        if load_future.result() is not None and load_future.result().ok:
            self.node.get_logger().info(f"Successfully loaded {controller_name}")
            # Configure and optionally start the controller after loading
            self.configure_controller(controller_name, should_start_controller)
        else:
            self.node.get_logger().error(f"Failed to load {controller_name}")

    def configure_controller(self, controller_name: str, should_start_controller: bool):
        # Configure the controller
        self.node.get_logger().info(f"Trying to configure controller {controller_name}")
        configure_request = ConfigureController.Request()
        configure_request.name = controller_name

        configure_future = self.configure_controller_client.call_async(configure_request)
        rclpy.spin_until_future_complete(self.node, configure_future)

        if configure_future.result() is not None and configure_future.result().ok:
            self.node.get_logger().info(f"Successfully configured {controller_name}")
            if should_start_controller:
                self.start_controller(controller_name)
        else:
            self.node.get_logger().error(f"Failed to configure {controller_name}")

    def start_controller(self, controller_name: str):
        # Start the controller
        self.node.get_logger().info(f"Trying to start controller {controller_name}")
        switch_request = SwitchController.Request()
        switch_request.activate_controllers = [controller_name]
        switch_request.deactivate_controllers = []
        switch_request.strictness = 1  # Strict mode

        switch_future = self.switch_controller_client.call_async(switch_request)
        rclpy.spin_until_future_complete(self.node, switch_future)

        if switch_future.result() is not None and switch_future.result().ok:
            self.node.get_logger().info(f"Successfully started {controller_name}")
        else:
            self.node.get_logger().error(f"Failed to start {controller_name}")

    def switch_controller(self, start_controller_name: str, stop_controller_name: str):
        # Switch controllers by activating one and deactivating another
        self.node.get_logger().info(f"Switching controllers: start {start_controller_name}, stop {stop_controller_name}")
        switch_request = SwitchController.Request()
        switch_request.activate_controllers = [start_controller_name]
        switch_request.deactivate_controllers = [stop_controller_name]
        switch_request.strictness = 1  # Strict mode

        switch_future = self.switch_controller_client.call_async(switch_request)
        rclpy.spin_until_future_complete(self.node, switch_future)

        if switch_future.result() is not None and switch_future.result().ok:
            self.node.get_logger().info(f"Successfully started {start_controller_name} and stopped {stop_controller_name}")
        else:
            self.node.get_logger().error(f"Failed to switch controllers")

    def list_controllers(self):
        # List all controllers
        self.node.get_logger().info("Listing controllers")
        list_future = self.list_controllers_client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self.node, list_future)

        if list_future.result() is not None:
            for controller in list_future.result().controller:
                self.node.get_logger().info(f"Controller: {controller.name}, State: {controller.state}")
        else:
            self.node.get_logger().error("Failed to list controllers")