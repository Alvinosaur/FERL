#! /usr/bin/env python
"""
This node demonstrates velocity-based PID control by moving the Jaco so that it
maintains a fixed distance to a target. Additionally, it supports human-robot
interaction in the form of online physical corrections.
The novelty is that it contains a protocol for elicitating new features from
human input provided to the robot.
"""
import torch
# import roslib
# roslib.load_manifest('kinova_demo')

import rospy
import math
import sys
import select
import os
import time

# import kinova_msgs.msg
# from kinova_msgs.srv import *
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float32, Float64MultiArray

from controllers.pid_controller import PIDController
from planners.trajopt_planner import TrajoptPlanner
from learners.phri_learner import PHRILearner
from utils import ros_utils, openrave_utils
from utils.environment import Environment
from utils.trajectory import Trajectory

import numpy as np
import pickle


def msg_to_pose(msg):
    pose = np.array([
        msg.pose.position.x,
        msg.pose.position.y,
        msg.pose.position.z,
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w,
    ])
    return pose


class FeatureElicitator():
    """
    This class represents a node that moves the Jaco with PID control AND supports receiving human corrections online.
    Additionally, it implements a protocol for elicitating novel features from human input.

    Subscribes to:
        /$prefix$/out/joint_angles	- Jaco sensed joint angles
        /$prefix$/out/joint_torques - Jaco sensed joint torques

    Publishes to:
        /$prefix$/in/joint_velocity	- Jaco commanded joint velocities
    """

    def __init__(self, environment):
        # Create ROS node.
        rospy.init_node("feature_elicitator")

        # Load parameters and set up subscribers/publishers.
        self.environment = environment
        self.load_parameters()
        self.register_callbacks()

        # Start admittance control mode.
        # ros_utils.start_admittance_mode(self.prefix)

        # Publish to ROS at 100hz.
        self.r = rospy.Rate(100)

        print("----------------------------------")
        print("Moving robot, type Q to quit:")

    def run(self):
        self.controller.path_end_T = None
        self.controller.path_start_T = None
        self.interacting = False
        while not rospy.is_shutdown() and self.controller.path_end_T is None:

            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = raw_input()
                if line == "q" or line == "Q" or line == "quit":
                    break
                elif line == "i":
                    self.interacting = not self.interacting
                    for _ in range(5):
                        self.is_intervene_pub.publish(
                            self.interacting or self.interaction_mode)  # True during intervention
                rospy.sleep(0.1)
            if self.curr_pos is None:
                continue
            self.cmd = np.clip(self.cmd, -0.5, 0.5)
            cmd_rad = (np.transpose(self.curr_pos) +
                       np.diag(self.cmd) * self.delta_T)[0]
            self.joints_deg_pub.publish(
                Float64MultiArray(data=180 * cmd_rad / np.pi))

            self.r.sleep()

        print("----------------------------------")
        # ros_utils.stop_admittance_mode(self.prefix)

    def load_parameters(self):
        """
        Loading parameters and setting up variables from the ROS environment.
        """
        # ----- General Setup ----- #
        self.delta_T = 0.1
        self.prefix = rospy.get_param("setup/prefix")
        pick = rospy.get_param("setup/start")
        place = rospy.get_param("setup/goal")
        self.start_joints_rad = np.array(pick) * (math.pi / 180.0)
        self.goal_joints_rad = np.array(place) * (math.pi / 180.0)
        self.goal_pose = None if rospy.get_param(
            "setup/goal_pose") == "None" else rospy.get_param("setup/goal_pose")
        self.T = rospy.get_param("setup/T")
        self.timestep = rospy.get_param("setup/timestep")
        self.save_dir = rospy.get_param("setup/save_dir")
        self.INTERACTION_TORQUE_THRESHOLD = rospy.get_param(
            "setup/INTERACTION_TORQUE_THRESHOLD")
        self.INTERACTION_TORQUE_EPSILON = rospy.get_param(
            "setup/INTERACTION_TORQUE_EPSILON")
        self.CONFIDENCE_THRESHOLD = rospy.get_param(
            "setup/CONFIDENCE_THRESHOLD")
        self.N_QUERIES = rospy.get_param("setup/N_QUERIES")
        self.nb_layers = rospy.get_param("setup/nb_layers")
        self.nb_units = rospy.get_param("setup/nb_units")

        # Openrave parameters for the environment.
        model_filename = rospy.get_param("setup/model_filename")
        LF_dict = rospy.get_param("setup/LF_dict")
        self.environment.setup(model_filename=model_filename, LF_dict=LF_dict)

        # ----- Planner Setup ----- #
        # Retrieve the planner specific parameters.
        planner_type = rospy.get_param("planner/type")
        if planner_type == "trajopt":
            max_iter = rospy.get_param("planner/max_iter")
            num_waypts = rospy.get_param("planner/num_waypts")

            # Initialize planner and compute trajectory to track.
            self.planner = TrajoptPlanner(
                max_iter, num_waypts, self.environment)
        else:
            raise Exception('Planner {} not implemented.'.format(planner_type))

        self.traj = self.planner.replan(
            self.start_joints_rad, self.goal_joints_rad, self.goal_pose, self.T, self.timestep)

        # Track if you have reached the start/goal of the path.
        self.reached_start = False
        self.reached_goal = False
        self.feature_learning_mode = False
        self.interaction_mode = False

        # Save the intermediate target configuration.
        self.curr_pos = None
        self.cur_ee_pose = None

        # Track data and keep stored.
        self.interaction_data = []
        self.interaction_time = []
        self.interaction_pose_traj = []
        self.feature_data = []
        self.track_data = False

        self.interacting = False

        # ----- Controller Setup ----- #
        # Retrieve controller specific parameters.
        controller_type = rospy.get_param("controller/type")
        if controller_type == "pid":
            # P, I, D gains.
            P = rospy.get_param("controller/p_gain") * np.eye(7)
            I = rospy.get_param("controller/i_gain") * np.eye(7)
            D = rospy.get_param("controller/d_gain") * np.eye(7)

            # Stores proximity threshold.
            epsilon = rospy.get_param("controller/epsilon")

            # Stores maximum COMMANDED joint torques.
            MAX_CMD = rospy.get_param("controller/max_cmd") * np.eye(7)

            self.controller = PIDController(P, I, D, epsilon, MAX_CMD)
        else:
            raise Exception(
                'Controller {} not implemented.'.format(controller_type))

        # Planner tells controller what plan to follow.
        self.controller.set_trajectory(self.traj)

        # Stores current COMMANDED joint torques.
        self.cmd = np.eye(7)

        # ----- Learner Setup ----- #
        constants = {}
        constants["step_size"] = rospy.get_param("learner/step_size")
        constants["P_beta"] = rospy.get_param("lea"
                                              "rner/P_beta")
        constants["alpha"] = rospy.get_param("learner/alpha")
        constants["n"] = rospy.get_param("learner/n")
        self.feat_method = rospy.get_param("learner/type")
        self.learner = PHRILearner(
            self.feat_method, self.environment, constants)

    def register_callbacks(self):
        """
        Sets up all the publishers/subscribers needed.
        """
        self.joints_deg_pub = rospy.Publisher(
            "/siemens_demo/joint_cmd", Float64MultiArray, queue_size=10)
        self.is_intervene_pub = rospy.Publisher(
            "/is_intervene", Bool, queue_size=10)
        # Create subscriber to joint_angles.
        rospy.Subscriber('/kinova/current_joint_state',
                         Float64MultiArray, self.joint_angles_callback, queue_size=1)
        # Create subscriber to joint torques
        rospy.Subscriber('/kinova/current_joint_torques',
                         Float64MultiArray, self.joint_torques_callback, queue_size=1)
        # subscriber for ee pose
        rospy.Subscriber('/kinova/pose_tool_in_base_fk',
                         PoseStamped, self.robot_pose_callback, queue_size=1)

    def robot_pose_callback(self, msg):
        self.cur_ee_pose = msg_to_pose(msg)

    def joint_angles_callback(self, msg):
        """
        Reads the latest position of the robot and publishes an
        appropriate torque command to move the robot to the target.
        """
        # Read the current joint angles from the robot.
        curr_pos = np.array(msg.data).reshape((7, 1)) * (math.pi / 180.0)

        self.curr_pos = curr_pos
        # Check if we are in feature learning mode.
        if self.feature_learning_mode:
            # Allow the person to move the end effector with no control resistance.
            self.cmd = np.zeros((7, 7))

            # If we are tracking feature data, update raw features and time.
            if self.track_data == True:
                # Add recording to feature data.
                self.feature_data.append(
                    self.environment.raw_features(curr_pos))
            return

        # When not in feature learning stage, update position.

        # Update cmd from PID based on current position.
        self.cmd = self.controller.get_command(self.curr_pos)

        # Check is start/goal has been reached.
        if self.controller.path_start_T is not None:
            self.reached_start = True
        if self.controller.path_end_T is not None:
            self.reached_goal = True

    def joint_torques_callback(self, msg):
        """
        Reads the latest torque sensed by the robot and records it for
        plotting & analysis
        """
        # Read the current joint torques from the robot.
        torque_curr = np.array(msg.data).reshape((7, 1))
        interaction = False
        # for i in range(7):
        #     # Center torques around zero.
        #     torque_curr[i][0] -= self.INTERACTION_TORQUE_THRESHOLD[i]
        #     # Check if interaction was not noise.
        #     if np.fabs(torque_curr[i][0]) > self.INTERACTION_TORQUE_EPSILON[i] and self.reached_start:
        #         interaction = True
        interaction = self.interacting
        # print("=======")
        # print(interaction)
        # print(self.interaction_mode)
        # print(len(self.interaction_data))
        if interaction:
            if True:  # self.reached_start and not self.reached_goal:
                timestamp = time.time() - self.controller.path_start_T
                self.interaction_data.append(torque_curr)
                self.interaction_pose_traj.append(self.cur_ee_pose.copy())
                self.interaction_time.append(timestamp)
                if self.interaction_mode == False:
                    self.interaction_mode = True
        else:
            if self.interaction_mode == True:
                #  save interaction data, traj, time
                if not os.path.exists("saved_data"):
                    os.makedirs("saved_data")

                saved_traj = np.load(
                    "/mnt/storage/catkin_ws/src/FERL/src/origial_traj.npz", allow_pickle=True)
                self.traj = Trajectory(
                    saved_traj['ee_pose_traj'], saved_traj['timesteps_traj'])
                self.traj_plan = self.traj.downsample(self.planner.num_waypts)

                np.savez("saved_data/interaction_datas.npz",
                         interaction_data=self.interaction_data,
                         interaction_pose_traj=self.interaction_pose_traj,
                         interaction_time=self.interaction_time,
                         traj_waypts=self.traj.waypts,
                         traj_waypts_time=self.traj.waypts_time)

                timestamp = time.time() - self.controller.path_start_T
                # Check if betas are above CONFIDENCE_THRESHOLD
                betas = self.learner.learn_betas(
                    self.traj, self.interaction_data[0], self.interaction_time[0])
                self.feature_learning_mode = True
                feature_learning_timestamp = time.time()
                input_size = len(self.environment.raw_features(torque_curr))
                self.environment.new_learned_feature(
                    self.nb_layers, self.nb_units)

                # Keep asking for input until we're happy.
                for query_i in range(self.N_QUERIES):
                    print "Need more data to learn the feature!"
                    self.feature_data = []

                    # Request the person to place the robot in a low feature value state.
                    print "Place the robot in a low feature value state and press ENTER when ready."
                    line = raw_input()
                    self.track_data = True

                    # Request the person to move the robot to a high feature value state.
                    print "Move the robot in a high feature value state and press ENTER when ready."
                    line = raw_input()
                    self.track_data = False

                    # Pre-process the recorded data before training.
                    feature_data = np.squeeze(np.array(self.feature_data))
                    lo = 0
                    hi = feature_data.shape[0] - 1
                    while np.linalg.norm(feature_data[lo] - feature_data[lo + 1]) < 0.01 and lo < hi:
                        lo += 1
                    while np.linalg.norm(feature_data[hi] - feature_data[hi - 1]) < 0.01 and hi > 0:
                        hi -= 1
                    print(feature_data)
                    feature_data = feature_data[lo:hi + 1, :]
                    print "Collected {} samples out of {}.".format(feature_data.shape[0], len(self.feature_data))

                    # Provide optional start and end labels.
                    start_label = 0.0
                    end_label = 1.0
                    print(
                        "Would you like to label your start? Press ENTER if not or enter a number from 0-10")
                    line = raw_input()
                    if line in [str(i) for i in range(11)]:
                        start_label = int(i) / 10.0

                    print(
                        "Would you like to label your goal? Press ENTER if not or enter a number from 0-10")
                    line = raw_input()
                    if line in [str(i) for i in range(11)]:
                        end_label = float(i) / 10.0

                    # Save newly collected data
                    np.savez("saved_data/feat_trace_%d.npz" % query_i,
                             start_label=[start_label],
                             end_label=[end_label],
                             processed_feat_data=feature_data,
                             orig_feat_data=self.feature_data
                             )

                    # Add the newly collected data.
                    self.environment.learned_features[-1].add_data(
                        feature_data, start_label, end_label)

                # Hold current pose while running adaptation
                for i in range(5):
                    self.joints_deg_pub.publish(
                        Float64MultiArray(data=180 * self.curr_pos / np.pi))
                rospy.sleep(0.1)
                self.is_intervene_pub.publish(False)

                # Train new feature with data of increasing "goodness".
                self.environment.learned_features[-1].train()

                # Compute new beta for the new feature.
                beta_new = self.learner.learn_betas(self.traj, torque_curr, timestamp, [
                                                    self.environment.num_features - 1])[0]
                betas.append(beta_new)

                # Move time forward to return to interaction position.
                self.controller.path_start_T += (time.time() -
                                                 feature_learning_timestamp)

                # We do not have misspecification now, so resume reward learning.
                self.feature_learning_mode = False

                # Learn reward.
                for i in range(len(self.interaction_data)):
                    self.learner.learn_weights(
                        self.traj, self.interaction_data[i], self.interaction_time[i], betas)

                self.traj = self.planner.replan(
                    self.start_joints_rad, self.goal_joints_rad, self.goal_pose, self.T, self.timestep, seed=self.traj_plan.waypts)
                self.traj_plan = self.traj.downsample(self.planner.num_waypts)
                self.controller.set_trajectory(self.traj)

                # Turn off interaction mode.
                self.interaction_mode = False
                self.interacting = False
                self.interaction_data = []
                self.interaction_pose_traj = []
                self.interaction_time = []


if __name__ == '__main__':
    ferl = FeatureElicitator()
    ferl.run()
