#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import Float64

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
import time
import threading

from tf2_msgs.msg import TFMessage
from std_msgs.msg import Float64MultiArray, Empty

from tf_transformations import euler_from_quaternion

# Global variables for the state
cart_pose_x = 0
cart_vel_x = 0
pole_pitch = 0
pole_tip_pose_z = 0

pause = True

# Q-learning parameters
steps = 1000
n_episodes = 300
div = 6
gamma = 0.99
alpha = 0.7
Q = np.random.uniform(-1, 1, (div**4, 2))


class QLearning(Node):
    def __init__(self):
        super().__init__('q_learning')

        self.results_ready = threading.Event()

        less_important_service_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST
        )

        important_service_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST
        )

        self.pub_cart = self.create_publisher(Float64MultiArray, '/joint_position_controller/commands', less_important_service_qos)
        self.pub_reset = self.create_publisher(Empty, '/cart_pole/reset', important_service_qos)

        self.loop_thread = threading.Thread(target=self.run_genetic_algorithm)
        self.loop_thread.start()

    def restart_cart_pole(self):
        """Reset the cart-pole system."""
        global cart_pose_x, cart_vel_x, pole_pitch, pole_tip_pose_z

        self.paused = True
        cart_pose_x = 0
        cart_vel_x = 0
        pole_pitch = 0
        pole_tip_pose_z = 1
       

        self.paused = True
              
        msg = Float64MultiArray()
        msg.data = [0.0]
        self.pub_cart.publish(msg)
                
        msg = Empty()
        self.pub_reset.publish(msg)
        self.pub_reset.publish(msg)
        self.pub_reset.publish(msg)
        
        self.paused = False

    def commander(self, episode):
        """Main Q-learning episode."""
        global cart_pose_x, pole_pitch, cart_vel_x, pole_tip_pose_z

        # Initialize variables
        observation = [0, 0, 0, 0]
        state = self.digit_state(observation)
        action = np.argmax(Q[state])
        reward_sum = 0
        pole_height_sum = 0
        fail = False
        epsilon = 1 / (episode + 1)
        time_interval = 0.02

        # Reset simulation
        self.restart_cart_pole()

        sim_steps = int(5 / time_interval)
        prev_pole_pitch = 0
        prev_cart_pose_x = 0

        for i in range(sim_steps):
            time1 = time.time()

            y_angular = (pole_pitch - prev_pole_pitch) / time_interval
            prev_pole_pitch = pole_pitch
            pole_height_sum += pole_tip_pose_z * time_interval

            cart_vel_x = (cart_pose_x - prev_cart_pose_x) / time_interval
            prev_cart_pose_x = cart_pose_x

            # Update observation
            observation[0] = cart_pose_x
            observation[1] = cart_vel_x
            observation[2] = pole_pitch
            observation[3] = y_angular

            self.get_logger().error(f"cart_pose_x: {cart_pose_x}")
            self.get_logger().error(f"cart_vel_x: {cart_vel_x}")
            self.get_logger().error(f"pole_pitch: {pole_pitch}")
            self.get_logger().error(f"y_angular: {y_angular}")

            # Reward function
            if abs(pole_pitch) > 0.5:
                reward_sum -= (sim_steps - i) * 10
                fail = True
            else:
                reward_sum += 10

            reward_sum -= abs(int(cart_pose_x * 10))

            # Q-learning update
            next_state = self.digit_state(observation)
            Q[state, action] += alpha * (reward_sum + gamma * max(Q[next_state]) - Q[state, action])

            # Exploration-exploitation trade-off
            action = np.argmax(Q[next_state]) if epsilon <= np.random.uniform(0, 1) else np.random.choice(2)

            # Apply force based on the action
            force = -8 if action == 0 else 8
            msg = Float64MultiArray()
            msg.data = [force]
            self.pub_cart.publish(msg)

            state = next_state

            if fail:
                break

            time2 = time.time()
            interval = time2 - time1
            if interval < time_interval:
                time.sleep(time_interval - interval)
        
        
        return reward_sum, pole_height_sum

    def stop(self):
        """Gracefully stop the node and thread."""
        self.running = False  # Stop the thread
        if self.loop_thread.is_alive():
            self.loop_thread.join()  # Ensure the thread is stopped before shutdown

    @staticmethod
    def reg1dim(x, y):
        """Simple linear regression to fit the rewards graph."""
        n = len(x)
        a = ((np.dot(x, y) - y.sum() * x.sum() / n) /
             ((x ** 2).sum() - x.sum() ** 2 / n))
        b = (y.sum() - a * x.sum()) / n
        return a, b

    @staticmethod
    def digit_state(observation):
        """Discretize the state space for Q-learning."""
        p, v, a, w = observation
        pn = np.digitize(p, np.linspace(-1.25, 1.25, div + 1)[1:-1])  # cart position
        vn = np.digitize(v, np.linspace(-2.5, 2.5, div + 1)[1:-1])  # cart velocity
        an = np.digitize(a, np.linspace(-0.625, 0.625, div + 1)[1:-1])  # pole angle
        wn = np.digitize
        wn = np.digitize(w, np.linspace(-5.0, 5.0, div + 1)[1:-1])  # pole angular velocity
        return pn + vn * div + an * div**2 + wn * div**3
    
    def run_genetic_algorithm(self):
        sum_rewards = 0
        pole_height_sum = 0
        rewards = []
        threshold = n_episodes - 100

        try:
            for episode in range(n_episodes):
                reward_sum, pole_time_height = self.commander(episode)
                self.get_logger().info(f"episode: {episode} reward_sum: {reward_sum} pole_height_sum: {pole_height_sum}")
                pole_height_sum += pole_time_height
                rewards.append(reward_sum)

                if episode >= threshold:
                    sum_rewards += reward_sum

            average_reward = sum_rewards / 100
            self.get_logger().info(f"Average reward: {average_reward}")
            self.get_logger().info(f"Pole time height: {pole_height_sum}")

            # Return data to be plotted later in the main thread
            # Signal that the computation is done
            self.rewards = rewards
            self.results_ready.set()  # Signal the main thread to plot the results
            self.get_logger().info("Genetic algorithm completed.")
           
        except Exception as e:
            if rclpy.ok():  # Log the exception only if the context is valid
                self.get_logger().error(f"Exception in genetic algorithm: {e}")
            else:
                self.get_logger().error(f"Exception during genetic algorithm execution: {e}")
                

class PoseSubscriber(Node):

    def __init__(self):
        super().__init__('pose_subscriber')
                # Define QoS profiles

        less_important_service_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST
        )

        self.cart_pole_subscription = self.create_subscription(
            TFMessage,
            '/model/cart_pole/pose',
            self.get_cart_pose,
            less_important_service_qos
        )
        self.get_logger().info('Subscribed to /model/cart_pole/pose')

    def get_cart_pose(self, data):
        """Callback to update cart and pole states from the pose topic."""
        global cart_pose_x, cart_vel_x, pole_pitch, pole_tip_pose_z

        
        for tfsf in data.transforms:
            if tfsf.child_frame_id == 'cart_pole/cart_link':
                cart_pose_x = tfsf.transform.translation.x
            elif tfsf.child_frame_id == 'cart_pole/pole_link':
                rpy_angles = euler_from_quaternion([tfsf.transform.rotation.x,
                                                    tfsf.transform.rotation.y,
                                                    tfsf.transform.rotation.z,
                                                    tfsf.transform.rotation.w])
                pole_pitch = rpy_angles[1]  # Extract pitch angle for pole
            elif tfsf.child_frame_id == 'cart_pole/tip_link':
                pole_tip_pose_z = tfsf.transform.translation.z  # Pole tip height

        # self.get_logger().info(f"pole_pitch: {pole_pitch}")
        # self.get_logger().info(f"pole z: {pole_tip_pose_z}")   

def plot_results(rewards):
        #  # Save Q-table to file
    with open('Qvalue.txt', 'w') as f:
        np.savetxt(f, Q)

    # Plot rewards over episodes
    fig, ax = plt.subplots()
    x = list(range(len(rewards)))
    a, b = QLearning.reg1dim(np.array(x), np.array(rewards))
    ax.set_ylim(-2400, 2400)
    ax.plot(x, rewards)
    ax.plot([0, max(x)], [b, a * max(x) + b])
    plt.show()

def main(args=None):
    rclpy.init(args=args)

    learning = QLearning()
    subscriber = PoseSubscriber()

    # Use MultiThreadedExecutor to allow concurrent execution of callbacks
    executor = MultiThreadedExecutor()
    executor.add_node(learning)
    executor.add_node(subscriber)

     # Spin the executor in its own thread
    spin_thread = threading.Thread(target=executor.spin)
    spin_thread.start()

    try:
        learning.results_ready.wait()  # Block until the genetic algorithm finishes
                
        plot_results(learning.rewards)  # Now we plot on the main thread
    except KeyboardInterrupt:
        pass
    finally:
        learning.stop()
        learning.destroy_node()
        subscriber.stop()
        subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()