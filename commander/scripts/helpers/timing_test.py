#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray, Empty
import time  # Import time module for timing

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')

        # Create a publisher to send messages
        self.pub_cart = self.create_publisher(Float64, '/cart_position', 10)

        # Timer period in seconds (e.g., 0.01 seconds = 100Hz)
        self.timer_period = 0.01

        # Create a timer to call the `timer_callback` function at the given interval
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # Initial state of the system
        self.cart_position = 0.0

        # Variable to store the time of the last callback
        self.last_callback_time = time.time()

    def timer_callback(self):
        # Calculate the time difference between the current and last callback
        current_time = time.time()
        duration = current_time - self.last_callback_time
        self.last_callback_time = current_time

        # Log the duration between callbacks
        self.get_logger().info(f'Time between callbacks: {duration:.6f} seconds')

        # Update cart position as an example
        self.cart_position += 0.035  # Example position change

        # Create and publish a message
        msg = Float64MultiArray()
        msg.data = [10]
        self.pub_cart.publish(msg)

        # Log the current cart position
        self.get_logger().info(f'Cart position: {self.cart_position}')

def main(args=None):
    rclpy.init(args=args)

    # Create the node
    my_node = MyNode()

    # Spin the node to handle callbacks
    rclpy.spin(my_node)

    # Shutdown the node after completion
    my_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()