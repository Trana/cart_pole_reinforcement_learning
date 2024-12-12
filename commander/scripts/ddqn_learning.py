#!/usr/bin/env python3

import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Float64MultiArray, Empty
from tf2_msgs.msg import TFMessage
from tf_transformations import euler_from_quaternion

from common.pose_subscriber import PoseSubscriber
from common.shared_state import SharedState
from common.training_config import TrainingConfig
from common.replay_buffer import ReplayBuffer
from common.fcq_network import FCQNetwork
from common.strategies.greedy_strategy import GreedyStrategy
from common.strategies.e_greedy_exp_strategy import EGreedyExpStrategy
import time  # Import time module for timing

class DQNLearning(Node):
    def __init__(self, shared_state, training_config):
        super().__init__('dqn_learning')

        # Set up publishers and threading
        self.shared_state = shared_state
        self.training_config = training_config
        self.results_ready = threading.Event()
        self.observations = []
        self.steps = []

        # Quality of Service Profiles
        less_important_service_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10
        )
        important_service_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )

        self.pub_cart = self.create_publisher(Float64MultiArray, '/joint_position_controller/commands', less_important_service_qos)
        self.pub_reset = self.create_publisher(Empty, '/cart_pole/reset', important_service_qos)

        self.episodes = 2000
        self.sim_steps = 100
        self.time_interval = 0.035
        self.is_training = True

        self.episode_running = False
        self.current_step = 0
        self.total_steps = 0
        self.current_episode = 1
        self.prev_pole_pitch = 0
        self.prev_cart_pose_x = 0
        self.pole_heights = []
        self.rewards = []
        self.currentEpisodeRewards = 0
        self.observations = []
        self.observation = np.zeros(4, dtype='float16')
        self.prev_action = 0
        self.prev_state = np.zeros(4, dtype='float16')

        
        self.num_states = 4
        self.num_actions = 2
        self.target_model = FCQNetwork(self.num_states, self.num_actions)
        self.online_model = FCQNetwork(self.num_states, self.num_actions)
        self.update_network()

        # self.value_optimizer = optim.Adam(self.online_model.parameters(), lr=self.training_config.learning_rate)
        self.value_optimizer = optim.RMSprop(self.online_model.parameters(), lr=self.training_config.learning_rate)       

        self.replay_buffer = ReplayBuffer(self.training_config.memory_capacity)

        self.training_strategy = EGreedyExpStrategy(init_epsilon=1.0,  
                                                      min_epsilon=0.3, 
                                                      decay_steps=(self.episodes *10))
        self.evaluation_strategy = GreedyStrategy()

        # self.epsilon = self.training_config.epsilon_start
        self.failed = False  # Initialize failure flag
        self.episode_start_time = time.time()
        # Create a timer to call the commander function every 0.02 seconds
        self.timer = self.create_timer(self.time_interval, self.timer_callback)
        
    def update_network(self):
        for target, online in zip(self.target_model.parameters(), 
                                    self.online_model.parameters()):
            target.data.copy_(online.data)

    def timer_callback(self):        

        
        if not self.episode_running or self.failed:
            self.get_logger().error(
                        f"Inside episode")
            if self.current_episode < self.episodes:
                if(self.current_step > 1):
                    self.get_logger().error(
                        f"Episode {self.current_episode-1} completed after {self.current_step} steps and reward {self.currentEpisodeRewards}")
                    self.get_logger().error(
                        f"Total steps {self.total_steps} current epsilon {self.training_strategy.epsilon}")
                    duration = time.time() - self.episode_start_time
                    self.get_logger().error(
                        f"Time per step {duration / self.current_step} time for episode {duration}")
                    
                    self.rewards.append(self.currentEpisodeRewards)
                    self.steps.append(self.current_step)

                self.get_logger().info(f"Starting episode {self.current_episode}")
                self.episode_running = True
                self.failed = False
                self.current_step = 0
                self.currentEpisodeRewards = 0
                self.current_episode += 1
                if(self.current_episode == self.episodes-200):
                    self.is_training = False
                self.get_logger().info(f"Is training {self.is_training}")
                self.restart_cart_pole()
                # Update target network
                if self.current_episode % self.training_config.target_update == 0:
                    self.update_network()   

                self.episode_start_time = time.time()

                # self.take_step(self.is_training)
               
                
            else:
                self.get_logger().info("Training completed.")
                self.results_ready.set()
                self.timer.cancel()
                return
        else:             

            if self.current_step < self.sim_steps:
                self.take_step(self.is_training)
                self.current_step += 1
            else:
                self.episode_running = False

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences
        batch_size = len(is_terminals)
        
        # argmax_a_q_sp = self.target_model(next_states).max(1)[1]
        argmax_a_q_sp = self.online_model(next_states).max(1)[1]

        q_sp = self.target_model(next_states).detach()
        max_a_q_sp = q_sp[
            np.arange(batch_size), argmax_a_q_sp].unsqueeze(1)
        target_q_sa = rewards + (self.training_config.gamma * max_a_q_sp * (1 - is_terminals))
        q_sa = self.online_model(states).gather(1, actions)

        td_error = q_sa - target_q_sa
        value_loss = td_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()        
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), 
                                       self.training_config.max_gradient_norm)
        self.value_optimizer.step()

    def take_step(self, is_training):
        # self.step(is_training)
        reward, pole_height, observation, failed = self.step(is_training)
        self.failed = failed
        self.observations.append(observation)
        self.pole_heights.append(pole_height)
        self.total_steps += 1
        self.currentEpisodeRewards += reward

    def get_current_state(self):

        state = np.zeros(4, dtype='float16')

        # pole_pitch = 
        # cart_pose_x = 

        # y_angular_vel = (pole_pitch - self.prev_pole_pitch) / self.time_interval if self.prev_pole_pitch != 0 else 0
        # cart_vel_x = (cart_pose_x - self.prev_cart_pose_x) / self.time_interval if self.prev_cart_pose_x != 0 else 0

        # self.prev_pole_pitch = pole_pitch
        # self.prev_cart_pose_x = cart_pose_x

        state[0] = self.shared_state.cart_pose_x
        state[1] = self.shared_state.cart_velocity_x
        state[2] = self.shared_state.pole_pitch
        state[3] = self.shared_state.pole_angular_velocity

        return state

    def step(self, is_training):
        
        done = False
        reward = 0

        next_state = self.get_current_state()

        if abs(self.shared_state.pole_pitch) > 0.6:
            # reward = -(self.sim_steps - self.current_step) * 5
            done = True
        elif self.current_step == 100:
            # reward = 200
            done = True
        else:
            reward = 1

        # Store prev state and action, and the reward and new state that it ended up in
        experience = (self.prev_state, self.prev_action, reward, next_state, float(done))
        self.replay_buffer.store(experience)
        self.prev_state = next_state

        if is_training:
            self.prev_action = self.training_strategy.select_action(self.online_model, next_state)
        else:
            self.prev_action = self.evaluation_strategy.select_action(self.online_model, next_state)

        # Publish control signal
        min_value = -0.02
        max_value = 0.02
        # action_min = 0
        # action_max = 9
        # position_change = ((self.prev_action - action_min) / (action_max - action_min)) * (max_value - min_value) + min_value
        if(self.prev_action==0):
            position_change = -0.03
        else:
            position_change = 0.03
        self.pub_cart.publish(Float64MultiArray(data=[self.shared_state.cart_pose_x + position_change]))

        pole_height = self.shared_state.pole_tip_pose_z * self.time_interval

        # Break if the episode is done
        if done:
            return reward, pole_height, next_state, True

        # If enough experience, sample and train
        if len(self.replay_buffer) > self.training_config.batch_size:
            experiences = self.replay_buffer.sample()
            experiences = self.online_model.load(experiences)
            self.optimize_model(experiences)      

        return reward, pole_height, next_state, False

    def restart_cart_pole(self):

        self.pub_cart.publish(Float64MultiArray(data=[0.0]))
        self.pub_cart.publish(Float64MultiArray(data=[0.0]))
        self.pub_reset.publish(Empty())
        self.pub_cart.publish(Float64MultiArray(data=[0.0]))

        # Wait for the cart to stabilize
        while abs(self.shared_state.cart_pose_x) > 0.005 or abs(self.shared_state.pole_pitch) > 0.005:
            pass
        

def plot_durations(steps):
    plt.figure(2)
    plt.clf()
    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(np.arange(len(steps)), steps)
    plt.pause(0.001)


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


def main(args=None):
    rclpy.init(args=args)

    shared_state = SharedState()
    traingin_config = TrainingConfig()    
    learning = DQNLearning(shared_state, traingin_config)
    subscriber = PoseSubscriber(shared_state)

    executor = MultiThreadedExecutor()
    executor.add_node(learning)
    executor.add_node(subscriber)

    spin_thread = threading.Thread(target=executor.spin)
    spin_thread.start()
    try:
        # Wait until the DQN algorithm has finished
        learning.results_ready.wait()

        plot_observation_distribution(learning, learning.observations)
        plot_durations(learning.steps)

        # Plot the rewards over episodes
        x = np.arange(0, len(learning.rewards))
        plt.plot(x, learning.rewards)
        plt.title("Rewards per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.show()

    except KeyboardInterrupt:
        pass
    finally:
        # Gracefully shut down
        learning.destroy_node()
        subscriber.destroy_node()
        rclpy.shutdown()
        spin_thread.join()


if __name__ == '__main__':
    main()