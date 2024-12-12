from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from tf_transformations import euler_from_quaternion

class PoseSubscriber(Node):
    def __init__(self, shared_state):
        super().__init__('pose_subscriber')

        self.shared_state = shared_state
        self.prev_cart_pose_x = None
        self.prev_pole_pitch = None
        self.prev_timestamp = None  # To store the previous timestamp

        self.cart_pole_subscription = self.create_subscription(
            TFMessage,
            '/model/cart_pole/pose',
            self.get_cart_pose,
            QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1
            )
        )

    def get_cart_pose(self, data):
        if not self.shared_state.paused:
            for tfsf in data.transforms:
                if tfsf.child_frame_id == 'cart_pole/cart_link':
                    self.shared_state.cart_pose_x = tfsf.transform.translation.x
                elif tfsf.child_frame_id == 'cart_pole/pole_link':
                    self.shared_state.pole_pitch = euler_from_quaternion([
                        tfsf.transform.rotation.x,
                        tfsf.transform.rotation.y,
                        tfsf.transform.rotation.z,
                        tfsf.transform.rotation.w
                    ])[1]  # Extract pitch angle for pole
                elif tfsf.child_frame_id == 'cart_pole/tip_link':
                    self.shared_state.pole_tip_pose_z = tfsf.transform.translation.z

            # Calculate velocities
            current_timestamp = data.transforms[0].header.stamp  # Assuming all transforms have the same timestamp
            current_time = current_timestamp.sec + current_timestamp.nanosec * 1e-9
            
            if self.prev_timestamp is not None:
                time_interval = current_time - self.prev_timestamp
                # self.get_logger().info(f"Current Timestamp: {current_time}")
                # self.get_logger().info(f"Cart Pose X: {self.shared_state.cart_pose_x}, Previous: {self.prev_cart_pose_x}")
                # self.get_logger().info(f"Pole Pitch: {self.shared_state.pole_pitch}, Previous: {self.prev_pole_pitch}")
                # self.get_logger().info(f"Time Interval: {time_interval}")
                if time_interval > 0:
                    # Ensure we have valid previous positions and calculate velocities
                    if self.prev_cart_pose_x is not None and self.prev_pole_pitch is not None:
                        cart_vel_x = (self.shared_state.cart_pose_x - self.prev_cart_pose_x) / time_interval
                        pole_angular_vel = (self.shared_state.pole_pitch - self.prev_pole_pitch) / time_interval

                        self.shared_state.cart_velocity_x = cart_vel_x
                        self.shared_state.pole_angular_velocity = pole_angular_vel
                        # self.get_logger().info(f"cart_velocity_x: {cart_vel_x}")
                        # self.get_logger().info(f"pole_angular_velocity: {pole_angular_vel}")
                else:
                    self.get_logger().warn("Skipped velocity calculation due to invalid time interval.")
            # Update previous values
            self.prev_cart_pose_x = self.shared_state.cart_pose_x
            self.prev_pole_pitch = self.shared_state.pole_pitch
            self.prev_timestamp = current_time