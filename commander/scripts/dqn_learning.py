#!/usr/bin/env python3

from collections import namedtuple
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import threading
import time

from commander.scripts.ddqn_learning import PoseSubscriber
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Float64MultiArray, Empty 
from tf2_msgs.msg import TFMessage
from tf_transformations import euler_from_quaternion

# Global variables for the state
cart_pose_x = 0
cart_vel_x = 0
pole_pitch = 0
pole_tip_pose_z = 0
paused = False


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
BATCH_SIZE = 32
CAPACITY = 10000
class ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = Transition(state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNLearning(Node):
    def __init__(self):
        super().__init__('dqn_learning')

        # Set up publishers and threading
        self.results_ready = threading.Event()
        self.paused = False
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

        self.gamma = 0.7
        self.r = 0.99
        self.lr = 0.001
        self.num_states = 4
        self.num_actions = 10
        self.episodes = 1000
        self.sim_steps = 200
        self.time_interval = 0.02  
        self.is_training = True
  
        self.episode_running = False
        self.current_step = 0
        self.current_episode = 1
        self.prev_pole_pitch = 0
        self.prev_cart_pose_x = 0
        self.pole_heights = []
        self.rewards  = []
        self.currentEpisodeRewards = 0
        self.observations = []
        self.observation = np.zeros(4, dtype='float16')
        self.agent = Agent(self.num_states, self.num_actions, self.gamma, self.r, self.lr)
        
        # Create a timer to call the commander function every 0.02 seconds
        self.timer = self.create_timer(self.time_interval, self.timer_callback)

    def timer_callback(self):
        global cart_pose_x, pole_pitch, cart_vel_x
        self.get_logger().info(f"self.agent.brain.eps {self.agent.brain.eps}")

        if not self.episode_running or self.failed:
            if self.current_episode < self.episodes:
                
                self.get_logger().error(f"Episode {self.current_episode-1} completed after {self.current_step} steps and reward {self.currentEpisodeRewards}")
                self.rewards.append(self.currentEpisodeRewards)
                self.steps.append(self.current_step)
                               
                self.get_logger().info(f"Starting episode {self.current_episode}")
                self.episode_running = True
                self.agent.brain.eps = max(0.01, 1 - 0.005 * self.current_episode)
                # self.agent.brain.eps = 1 / (self.current_episode + 1)
                self.currentEpisodeRewards = 0 
                self.current_step = 0              
                self.restart_cart_pole()

                self.take_step(self.agent, self.is_training)

                self.current_episode += 1
                

            else:
                self.get_logger().info("Training completed.")
                self.results_ready.set()
                self.timer.cancel()
                return

        if self.current_step < self.sim_steps:
            self.take_step(self.agent, self.is_training)
            self.current_step += 1
        else:
            self.episode_running = False

    def take_step(self, agent, is_training):
        reward, pole_height, observation, failed = self.step(agent, is_training)
        self.failed = failed
        self.observations.append(observation)
        self.pole_heights.append(pole_height)
        self.currentEpisodeRewards += reward

    def step(self, agent, is_training):
        global cart_pose_x, pole_pitch, cart_vel_x, pole_tip_pose_z
        
        failed = False
        observations = []
        reward = 0 
        next_obs = np.zeros(4, dtype='float16')
    
        y_angular_vel = (pole_pitch - self.prev_pole_pitch) / self.time_interval if self.prev_pole_pitch != 0 else 0
        cart_vel_x = (cart_pose_x - self.prev_cart_pose_x) / self.time_interval if self.prev_cart_pose_x != 0 else 0

        next_obs[0] = cart_pose_x
        next_obs[1] = cart_vel_x
        next_obs[2] = pole_pitch
        next_obs[3] = y_angular_vel
        observations.append(next_obs.copy())

        self.prev_pole_pitch = pole_pitch
        self.prev_cart_pose_x = cart_pose_x

        action = agent.getAction(self.observation, is_training)

        # Determine if episode is done
        # if abs(pole_pitch) > 0.6 or self.current_step == self.sim_steps - 1:
        #     failed = True
        #     reward = -200 if self.current_step < self.sim_steps - 1 else self.currentEpisodeRewards
        # else:
        #     reward = 6 - abs(pole_pitch) * 10

        # if abs(pole_pitch) > 0.6:
        #     failed = True
        #     reward = -100
        # else:
        #     reward = 1

        if(abs(pole_pitch) > 0.6):
            reward = -(self.sim_steps-self.current_step)*10
            failed = True
        else:
            reward = 10

        # Update Q network during training
        if is_training:
            agent.updateQnet(self.observation.copy(), action, reward, next_obs)

        # Publish control signal
        min_value = -0.08
        max_value = 0.08
        action_min = 0
        action_max = 9  # Assuming action can be 0, 1, or 2


        position_change = ((action - action_min) / (action_max - action_min)) * (max_value - min_value) + min_value
        # self.get_logger().error(f"action {action} cart_pose_x {cart_pose_x} pole_pitch {pole_pitch}  position_change {position_change} ")
            
        # self.get_logger().error(f'action: {action} position_change {position_change}')
        self.pub_cart.publish(Float64MultiArray(data=[cart_pose_x + position_change]))

        self.observation = next_obs.copy()
        pole_height = pole_tip_pose_z * self.time_interval

        # Break if the episode is done
        if failed:
            return reward, pole_height, next_obs, True                

        return reward, pole_height, next_obs, False 

    def restart_cart_pole(self):
        global cart_pose_x, cart_vel_x, pole_pitch, pole_tip_pose_z, paused
       
      
        self.pub_cart.publish(Float64MultiArray(data=[0.0]))
        self.pub_reset.publish(Empty())
        
        ready=False
        while abs(cart_pose_x) > 0.005  or abs(pole_pitch) > 0.005:
            ready=True
            
class QNet(nn.Module):
    def __init__(self, num_states, num_actions):
        super(QNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_states, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.fc(x)


class Brain:
    def __init__(self, num_states, num_actions, gamma, r, lr):
        self.eps = 1.0
        self.gamma = gamma
        self.r = r

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.q_net = QNet(num_states, num_actions).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def updateQnet(self, obs_numpy, action, reward, next_obs_numpy):
        obs_tensor = torch.from_numpy(obs_numpy).float().unsqueeze(0).to(self.device)
        next_obs_tensor = torch.from_numpy(next_obs_numpy).float().unsqueeze(0).to(self.device)

        self.optimizer.zero_grad()

        q_values = self.q_net(obs_tensor)
        with torch.no_grad():
            next_q_values = self.q_net(next_obs_tensor)
            target_q_values = q_values.clone()
            target_q_values[:, action] = reward + self.gamma * next_q_values.max(1)[0].item()

        loss = self.criterion(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

    def getAction(self, obs_numpy, is_training):
        if is_training and np.random.rand() < self.eps:
            action = np.random.randint(self.q_net.fc[-1].out_features)
        else:
            obs_tensor = torch.from_numpy(obs_numpy).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(obs_tensor)
                action = torch.argmax(q_values, dim=1).item()

        # if is_training and self.eps > 0.1:
        #     self.eps *= self.r         
        return action


class Agent:
    def __init__(self, num_states, num_actions, gamma, r, lr):
        self.brain = Brain(num_states, num_actions, gamma, r, lr)

    def updateQnet(self, obs, action, reward, next_obs):
        self.brain.updateQnet(obs, action, reward, next_obs)

    def getAction(self, obs, is_training):
        return self.brain.getAction(obs, is_training)

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

    learning = DQNLearning()
    subscriber = PoseSubscriber()

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
        # Plot the results
        x = np.arange(0, len(learning.rewards))
        plt.plot(x, learning.rewards)
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