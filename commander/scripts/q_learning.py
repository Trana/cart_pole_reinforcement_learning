#!/usr/bin/env python3

import os
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

paused = False

class QLearning(Node):
    def __init__(self):
        super().__init__('q_learning')
                
        self.results_ready = threading.Event()

        # Publishers
        self.pub_cart = self.create_publisher(Float64MultiArray, '/joint_position_controller/commands', 10)
        self.pub_reset = self.create_publisher(Empty, '/cart_pole/reset', 10)

        # Episode settings
        self.current_episode = 1
        self.max_episodes = 300
        self.time_interval = 0.02  # 50 Hz or every 0.02 seconds
        # self.last_callback_time = time.time()   
        
        # Create a timer to call the commander function every 0.02 seconds
        self.timer = self.create_timer(self.time_interval, self.timer_callback)

        # Initialize Q-learning parameters
        self.episode_running = False
        self.sim_steps = 200  # Total simulation steps per episode
        self.current_step = 0

        self.nrOfBins = 6
        self.gamma = 0.99
        self.alpha = 0.3
        self.Q = self.load_q_table()

        self.prev_pole_pitch = 0
        self.prev_cart_pose_x = 0

        self.observations = []
        self.pole_heights = []
        self.rewards  = []
        self.currentEpisodeRewards = 0

        self.state = self.digit_state([0,0,0,0])
        self.action = np.argmax(self.Q[self.state])
        self.failed = False
        self.steps = []


    def timer_callback(self):
        global cart_pose_x, pole_pitch, cart_vel_x
        """The timer callback function that is triggered every 0.02 seconds"""
        # Calculate the time difference between the current and last callback
        # current_time = time.time()
        # duration = current_time - self.last_callback_time
        # self.last_callback_time = current_time
        # self.get_logger().info(f'Time between callbacks: {duration:.6f} seconds')
        
        if not self.episode_running or self.failed:
            if self.current_episode < self.max_episodes:
                self.get_logger().error(f"Episode {self.current_episode-1} completed after {self.current_step} steps and reward {self.currentEpisodeRewards}")
                self.rewards.append(self.currentEpisodeRewards)
                self.currentEpisodeRewards = 0                
                self.get_logger().info(f"Starting episode {self.current_episode}")
                self.episode_running = True
                self.steps.append(self.current_step)
                self.current_step = 0  # Reset step counter
              
                self.state = self.digit_state([0,0,0,0])
                self.action = np.argmax(self.Q[self.state])
                
                self.restart_cart_pole()
                self.get_logger().error(f"cart_pose_x {cart_pose_x} cart_vel_x {cart_vel_x} pole_pitch {pole_pitch} pole_pitch {pole_pitch}")

                reward, pole_height, observation, failed = self.step(self.current_episode)
                self.failed = failed
                self.observations.append(observation)
                self.pole_heights.append(pole_height)
                self.currentEpisodeRewards += reward
                self.current_episode += 1

            else:
                self.get_logger().info("Training completed.")
                self.results_ready.set()
                self.timer.cancel()
                return

        if self.current_step < self.sim_steps:
            # Execute commander steps
            reward, pole_height, observation, failed = self.step(self.current_episode)
            self.failed = failed
            self.observations.append(observation)
            self.pole_heights.append(pole_height)
            self.currentEpisodeRewards += reward
            # self.get_logger().error(f"rewards {self.rewards} reward {self.currentEpisodeRewards}")

            self.current_step += 1
        else:
            # End the episode
            
            self.episode_running = False

    def step(self, episode):
        """Main Q-learning episode executed in steps"""
        global pole_tip_pose_z

        self.failed = False
        reward = 0

        epsilon = 1 / (episode + 1)

        observation = self.get_current_observations()

        if(abs(pole_pitch) > 0.4):
            reward -= (self.sim_steps-self.current_step)*10
            self.failed = True
        else:
            reward = 10

        reward -= abs(int(cart_pose_x*10))
        
        # Q-learning update
        next_state = self.digit_state(observation)
        self.Q[self.state, self.action] += self.alpha * (reward + self.gamma * max(self.Q[next_state]) - self.Q[self.state, self.action])
      
        # Action selection
        self.action = np.argmax(self.Q[next_state]) if epsilon <= np.random.uniform(0, 1) else np.random.choice(2)

        positionChange = -0.04 if self.action == 0 else 0.04

        self.publish_cart_position(cart_pose_x + positionChange)

        self.state = next_state 
        pole_height = pole_tip_pose_z * self.time_interval

        if self.failed:
            return reward, pole_height, observation, True

        return reward, pole_height, observation, False

    def get_current_observations(self):
        global cart_pose_x, pole_pitch, cart_vel_x
        observation = [0, 0, 0, 0]
        y_angular_vel = 0

        if self.prev_pole_pitch != 0:
            y_angular_vel = (pole_pitch - self.prev_pole_pitch) / self.time_interval  

        self.prev_pole_pitch = pole_pitch

        cart_vel_x = 0
        if self.prev_cart_pose_x != 0:
            cart_vel_x = (cart_pose_x - self.prev_cart_pose_x) / self.time_interval

        self.prev_cart_pose_x = cart_pose_x

        observation[0] = cart_pose_x
        observation[1] = cart_vel_x
        observation[2] = pole_pitch
        observation[3] = y_angular_vel

        return observation

    def publish_cart_position(self, position):
        msg = Float64MultiArray()
        msg.data = [position]
        self.pub_cart.publish(msg)

    def digit_state(self, observation):
        p, v, a, w = observation
        pn = np.digitize(p, np.linspace(-1.0, 1.0, self.nrOfBins + 1)[1:-1])  # cart position
        vn = np.digitize(v, np.linspace(-5.0, 5.0, self.nrOfBins + 1)[1:-1])  # cart velocity
        an = np.digitize(a, np.linspace(-0.5, 0.5, self.nrOfBins + 1)[1:-1])  # pole angle
        wn = np.digitize
        wn = np.digitize(w, np.linspace(-10.0, 10.0, self.nrOfBins + 1)[1:-1]) # pole angular velocity
        return pn + vn * self.nrOfBins + an * self.nrOfBins**2 + wn * self.nrOfBins**3

        # Function to load the Q-table
    def load_q_table(self, filename='Qvalue.txt2'):
        if os.path.exists(filename):
            self.get_logger().error(f"Loading Q-table from {filename}")
            return np.loadtxt(filename)
        else:
            self.get_logger().error(f"No Q-table found, initializing new Q-table.")
            return np.random.uniform(-1, 1, (self.nrOfBins**4, 2))  # Initialize Q-table if file doesn't exist

    def restart_cart_pole(self):
        """Reset the cart-pole system."""
        global cart_pose_x, cart_vel_x, pole_pitch, pole_tip_pose_z, paused

        paused = True
        cart_pose_x = 0
        cart_vel_x = 0
        pole_pitch = 0
        pole_tip_pose_z = 1

        msg = Float64MultiArray()
        msg.data = [0.0]
        self.pub_cart.publish(msg)

        msg = Empty()
        self.pub_reset.publish(msg)
        paused = False

        ready=False
        while abs(cart_pose_x) > 0.005  or abs(pole_pitch) > 0.005:
            ready=True

    @staticmethod
    def reg1dim(x, y):
        """Simple linear regression to fit the rewards graph."""
        n = len(x)
        a = ((np.dot(x, y) - y.sum() * x.sum() / n) /
             ((x ** 2).sum() - x.sum() ** 2 / n))
        b = (y.sum() - a * x.sum()) / n
        return a, b

class PoseSubscriber(Node):

    def __init__(self):
        super().__init__('pose_subscriber')
        
        less_important_service_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
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
        global cart_pose_x, cart_vel_x, pole_pitch, pole_tip_pose_z, paused

        if not paused:
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
    # Plot rewards over episodes
    fig, ax = plt.subplots()
    x = list(range(len(rewards)))
    a, b = QLearning.reg1dim(np.array(x), np.array(rewards))
    ax.set_ylim(-2000, 2000)
    ax.plot(x, rewards)
    ax.plot([0, max(x)], [b, a * max(x) + b])
    plt.show()

def plot_observation_distribution(node, observations):
    positions = []
    velocities = []
    angles = []
    angular_velocities = []

    for obs in observations:
        positions.append(obs[0])  # cart position
        velocities.append(obs[1])  # cart velocity
        angles.append(obs[2])  # pole angle
        angular_velocities.append(obs[3])  # pole angular velocity

    # Plot histograms of the observations
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 3, 1)
    plt.xlim(-20, 20)
    plt.hist(positions)
    plt.title("Cart Position Distribution")

    plt.subplot(3, 3, 2)
    plt.hist(velocities)
    plt.title("Cart Velocity Distribution")

    plt.subplot(3, 3, 3)
    plt.hist(angles)
    plt.title("Pole Angle Distribution")

    plt.subplot(3, 3, 4)
    plt.hist(angular_velocities)
    plt.title("Pole Angular Velocity Distribution")

    plt.subplot(3, 3, 5)
    plt.plot([obs[2] for obs in observations])  # Pole pitch is observation[2]
    plt.title("Raw Pole Pitch Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Pole Pitch (radians)")

    plt.tight_layout()
    # plt.show()

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
        plot_observation_distribution(learning, learning.observations)
        plot_results(learning.rewards)  # Now we plot on the main thread
        #  # Save Q-table to file
        with open('Qvalue.txt', 'w') as f:
            np.savetxt(f, learning.Q)
        x = np.arange(0, len(learning.steps))
        plt.plot(x, learning.steps)
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        learning.destroy_node()
        subscriber.stop()
        subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()