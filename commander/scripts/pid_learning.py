#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
import random
import time
import datetime
import threading

from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Empty

from tf_transformations import euler_from_quaternion

from deap import base, creator, tools

cart_pose = Pose()
pole_pose = Pose()
pole_twist = Twist()
y_rotation = 0
cart_pose_x = 0
pole_tip_pose_z = 0

class PIDGainOptimizer(Node):
    def __init__(self):
        super().__init__('optimize_pid_learning')
        
        qos_profile = QoSProfile(depth=10)
        self.cart_pole_subscription = self.create_subscription(
            TFMessage,
            '/model/cart_pole/pose',
            self.get_cart_pose,
            qos_profile
        )

        self.paused = False
        self.get_logger().info('Subscribed to /model/cart_pole/pose')

        self.pub_cart = self.create_publisher(Float64MultiArray, '/joint_position_controller/commands', qos_profile)
        self.pub_reset = self.create_publisher(Empty, '/cart_pole/reset', qos_profile)
      

    def get_cart_pose(self, data):
        global cart_pose
        global pole_pose
        global pole_twist 
        global y_rotation 
        global cart_pose_x 
        global pole_tip_pose_z

        if not self.paused:
            for tfsf in data.transforms:
                
                if(tfsf.child_frame_id == 'cart_pole/cart_link'):
                    cart_pose = tfsf.transform.translation
                    cart_pose_x = cart_pose.x
                if(tfsf.child_frame_id == 'cart_pole/pole_link'):
                    rpy_angels = euler_from_quaternion([tfsf.transform.rotation.x, tfsf.transform.rotation.y, tfsf.transform.rotation.z, tfsf.transform.rotation.w])
                    # self.get_logger().info(f"rpy_angels: {rpy_angels}")
                    y_rotation = rpy_angels[1]
                if(tfsf.child_frame_id == 'cart_pole/tip_link'):
                    pole_tip_pose = tfsf.transform.translation
                    pole_tip_pose_z = pole_tip_pose.z         

        # self.get_logger().info(f"y_rotation: {y_rotation}")
        # self.get_logger().info(f"pole z: {pole_tip_pose_z}")   

    def restart_cart_pole(self):
        global cart_pose
        global pole_pose
        global pole_twist
        global y_rotation
        global cart_pose_x
        global pole_tip_pose_z

        
        cart_pose = Pose()
        pole_pose = Pose()
        pole_twist = Twist()
        y_rotation = 0
        cart_pose_x = 0
        pole_tip_pose_z = 1

        msg = Float64MultiArray()
        msg.data = [0.0]
        self.pub_cart.publish(msg)
        
        msg = Empty()
        self.pub_reset.publish(msg)
        
        while abs(cart_pose_x) > 0.005  or abs(y_rotation) > 0.005:
            _=True

    def gain_evaluation(self, individual):
        global cart_pose
        global pole_pose
        global pole_twist 
        global y_rotation 
        global cart_pose_x 
        global pole_tip_pose_z

        self.get_logger().info(f"Evaluating individual: {individual}")

        Kp_y = individual[0]
        Ki_y = individual[1] 
        Kd_y = individual[2] 

        Kp_p = individual[3]
        Ki_p = individual[4]
        Kd_p = individual[5]

        time_interval = 0.02

        # Reset simulation
        self.restart_cart_pole()

        # Reset values
        target_yaw_angle = 0
        target_cart_pose_x = 0
        last_error_yaw = 0
        last_error_pos = 0
        integral_position_error = 0
        integral_yaw_error = 0
        pole_tip_height_sum = 0
        x_displacement_sum = 0
            
        for _ in range(1000):
            time1 = time.time()

            # Increase jay angle error by 10 to get it more in line with the size of pos errors
            error_yaw = (target_yaw_angle-y_rotation) * 20

            integral_yaw_error += ((error_yaw + last_error_yaw)*time_interval/2)
            derivative_jaw_error = ((error_yaw - last_error_yaw)/time_interval)

            effort_yaw = (Kp_y*error_yaw) + (Ki_y*integral_yaw_error) + (Kd_y* derivative_jaw_error) 

            # cart x
            pole_tip_height_sum += pole_tip_pose_z
            error_pos = target_cart_pose_x - cart_pose_x
            integral_position_error += (error_pos + last_error_pos)*time_interval/2
            x_displacement_sum += abs(cart_pose_x)
            effort_pos = (Kp_p*error_pos  + 
                        Ki_p*integral_position_error +
                        Kd_p*(error_pos - last_error_pos)/time_interval)   

            effort = effort_yaw + effort_pos    
            
            # self.get_logger().info(f'error_yaw: {error_yaw}')
            # self.get_logger().info(f'last_error_yaw: {last_error_yaw}')
            # self.get_logger().info('----')
            # self.get_logger().info(f'error_pos: {error_pos}')
            # self.get_logger().info(f'last_error_pos: {last_error_pos}')
            # self.get_logger().info('----')
            # self.get_logger().info(f'effort_yaw: {effort_yaw}')
            # self.get_logger().info(f'effort_pos: {effort_pos}')
            # self.get_logger().info('----')            
            # self.get_logger().info(f'Published velocity command: {msg.data}')
            
            last_error_yaw = error_yaw
            last_error_pos = error_pos

            msg = Float64MultiArray()
            msg.data = [effort]
            self.pub_cart.publish(msg)

            time2 = time.time()
            interval = time2 - time1
            if interval < time_interval:
                time.sleep(time_interval - interval)

        integral_position_error = 0
        integral_yaw_error = 0

        self.get_logger().info(f"x_displacement_sum: {x_displacement_sum}, pole_tip_height_sum: {pole_tip_height_sum}")  

        return x_displacement_sum, pole_tip_height_sum  

def run_genetic_algorithm(node):
    # Genetic algorithm setup
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_gene", random.uniform, 0, 20)
    toolbox.register("evaluate", node.gain_evaluation)
    toolbox.register("mate", tools.cxBlend, alpha=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # GA parameters
    N_GEN = 12        # number of generations
    POP_SIZE = 6    # number of individuals
    CX_PB = 0.5      # crossover probability
    MUT_PB = 0.3     # mutation probability

    points_number = 6
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_gene, points_number)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Create population
    pop = toolbox.population(n=POP_SIZE)

    # Evaluate each individual
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    g = 0
    while g < N_GEN:
        g += 1
        node.get_logger().info(f"-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUT_PB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population
        pop[:] = offspring

        fits0 = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits0) / length
        sum2 = sum(x * x for x in fits0)
        std = abs(sum2 / length - mean**2)**0.5

        node.get_logger().info(f"  Min %s" % min(fits0))
        node.get_logger().info(f"  Max %s" % max(fits0))
        node.get_logger().info(f"  Avg %s" % mean)
        node.get_logger().info(f"  Std %s" % std)

    # Choosing the best individual
    best_ind = tools.selBest(pop, 1)[0]
    node.get_logger().info(f"Best individual is %s, %s" % (best_ind, best_ind.fitness.values))


def main(args=None):
    rclpy.init(args=args)
    node = PIDGainOptimizer()

    # Run the genetic algorithm in a separate thread to not block subscription
    ga_thread = threading.Thread(target=run_genetic_algorithm, args=(node,))
    ga_thread.start()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()