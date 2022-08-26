import torch
import os
import numpy as np
import math
from .learned_feature import LearnedFeature
from kinova_robot import KinovaRobot   # NOTE: symlink with opa/kinova_robot.py
import sys
sys.path.append("/usr/local/lib/openrave0.53-plugins")
import openravepy
from openravepy import *

from openrave_utils import *
from learned_feature import LearnedFeature


class EnvironmentBase(object):
	"""
	This class creates an OpenRave environment and contains all the
	functionality needed for custom features and constraints.
	"""

	def __init__(self, use_predefined_feats):

		self.kinova_robot = KinovaRobot()

		# Populated by child class specific init()
		# TODO: try out efficiency
		self.feature_func_list = []
		self.feature_list = []
		self.weights = []
		self.object_poses = dict()

		self.use_predefined_feats = use_predefined_feats
		if use_predefined_feats:
			self.feature_func_list.append(self.dist_to_objects_feat)
			self.feature_list.append("dist_to_objects")
			self.weights.append(1.0)

		# Create a list of learned features.
		self.learned_features = []

	def setup(self, model_filename, viewer=True, LF_dict=None):
		# Insert any objects you want into environment.
		self.env, self.robot = initialize(model_filename, viewer=viewer)

		# Initialize LF_dict optionally for learned features.
		self.LF_dict = LF_dict

	# -- Compute features for all waypoints in trajectory. -- #
	def featurize(self, waypts, feat_idxs=None):
		"""
		Computes the features for a given trajectory.
		---
        Params:
            waypts -- trajectory waypoints
            feat_idx -- list of feature indices (optional)
        Returns:
            features -- list of feature values (T x num_features)
		"""
		# if no list of idx is provided use all of them
		if feat_idxs is None:
			feat_idxs = list(np.arange(len(self.feature_list)))

		# precompute fk poses to avoid repeat in each feature func
		pose_waypts = [
			self.kinova_robot.fk(waypt)[1] for waypt in waypts
		]

		T = len(waypts)
		all_features = []
		for feat_idx in feat_idxs:
			feat_name = self.feature_list[feat_idx]

			features_per_wpt = []
			for wpt_idx in range(T - 1):
				if feat_name == 'efficiency':
					waypt = np.concatenate((waypts[wpt_idx + 1], waypts[wpt_idx]))
					pose_waypt = np.concatenate(
						(pose_waypts[wpt_idx + 1], pose_waypts[wpt_idx]))
				else:
					waypt = waypts[wpt_idx]
					pose_waypt = pose_waypts[wpt_idx]

				# either single scalar or vector of (n_objs, )
				features_per_wpt.append(self.featurize_single(waypt, pose_waypt, feat_idx))

			# convert to array, either (1 x T) or (n_objs x T) if feature func is per object
			features_per_wpt = np.array(features_per_wpt)
			if len(features_per_wpt.shape) == 1:
				features_per_wpt = features_per_wpt[np.newaxis]

			all_features.append(features_per_wpt)

		# (num_features x T) -> (T x num_features)
		all_features = np.vstack(all_features).T
		return all_features

	# -- Compute single feature for single waypoint -- #
	def featurize_single(self, waypt, pose_waypt, feat_idx):
		"""
		Computes given feature value for a given waypoint.
		---
        Params:
            waypt -- single waypoint (joints)
			pose_waypt -- single waypoint (pose_quat)
            feat_idx -- feature index
        Returns:
            featval -- feature value
		"""
		# If it's a learned feature, feed in raw_features to the NN.
		if self.feature_list[feat_idx] == 'learned_feature':
			waypt = self.raw_features(waypt)
		# Compute feature value.
		featval = self.feature_func_list[feat_idx](waypt, pose_waypt)
		if self.feature_list[feat_idx] == 'learned_feature':
			featval = featval[0][0]

		return featval

	# -- Return raw features -- #
	def raw_features(self, waypt):
		"""
		Computes raw state space features for a given waypoint.
		---
        Params:
            waypt -- single waypoint
        Returns:
            raw_features -- list of raw feature values
		"""
		object_coords = np.vstack([v for v in self.object_poses.values()])
		Tall = self.get_torch_transforms(waypt)
		coords = Tall[:, :3, 3]
		orientations = Tall[:, :3, :3]
		object_coords = torch.from_numpy(object_coords)
		return torch.reshape(torch.cat((waypt.squeeze(), orientations.flatten(), coords.flatten(), object_coords.flatten())), (-1,))

	def get_torch_transforms(self, waypt):
		"""
		Computes torch transforms for given waypoint.
		---
        Params:
            waypt -- single waypoint
        Returns:
            Tall -- Transform in torch for every joint (7D)
		"""
		joint_poses, ee_pose = self.robot.fk(waypt)
		Tall = torch.from_numpy(np.vstack(joint_poses))

		return Tall

	# -- Update object pose to handle moving objects -- #
	def update_object_pose(self, key, pose_quat):
		assert key in self.object_poses, "Key " + key + " not found in object_poses"
		self.object_poses[key] = pose_quat

	# -- Instantiate a new learned feature -- #

	def new_learned_feature(self, nb_layers, nb_units, checkpoint_name=None):
		"""
		Adds a new learned feature to the environment.
		--
		Params:
			nb_layers -- number of NN layers
			nb_units -- number of NN units per layer
			checkpoint_name -- name of NN model to load (optional)
		"""
		self.learned_features.append(
			LearnedFeature(nb_layers, nb_units, self.LF_dict))
		self.feature_list.append('learned_feature')
		# initialize new feature weight with zero
		self.weights = np.hstack((self.weights, np.zeros((1, ))))

		# If we can, load a model instead of a blank feature.
		if checkpoint_name is not None:
			here = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
			self.learned_features[-1] = torch.load(here +
                                          '/data/final_models/' + checkpoint_name)

		self.feature_func_list.append(self.learned_features[-1].function)

	# -- Efficiency -- #

	def efficiency_features(self, waypt):
		"""
		Computes efficiency feature for waypoint, confirmed to match trajopt.
		---
        Params:
            waypt -- single waypoint
        Returns:
            dist -- scalar feature
		"""

		return np.linalg.norm(waypt[:7] - waypt[7:])**2

	# -- Distance to Robot Base (origin of world) -- #

	def dist_to_objects_feat(self, waypt, pose_waypt):
		object_coords = np.vstack([v for v in self.object_poses.values()])
		dists = np.linalg.norm(object_coords[:, 0:3] - pose_waypt[0:3], axis=-1)
		# NOTE: multiply -1 because features used in a "reward" function
		# learned features describe "rewards"
		return -dists

	# ---- Custom environmental constraints --- #
	def ee_constraint(self, waypt, pose_waypt):
		"""
		Constrains z-axis of robot's end-effector to always be above the table.
		"""
		if pose_waypt[2] > -0.1 and pose_waypt[2] < 0.35:
			return 0
		return 10000

	# ---- Helper functions ---- #

	def update_curr_pos(self, curr_pos):
		"""
		Updates DOF values in OpenRAVE simulation based on curr_pos.
		----
		curr_pos - 7x1 vector of current joint angles (degrees)
		"""
		pos = np.array([curr_pos[0][0], curr_pos[1][0], curr_pos[2][0] + math.pi,
		               curr_pos[3][0], curr_pos[4][0], curr_pos[5][0], curr_pos[6][0], 0, 0, 0])

		self.robot.SetDOFValues(pos)

	def kill_environment(self):
		"""
		Destroys openrave thread and environment for clean shutdown.
		"""
		self.env.Destroy()
		RaveDestroy()  # destroy the runtime
