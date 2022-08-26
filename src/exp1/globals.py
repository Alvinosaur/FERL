import sys
import numpy as np
import torch

from scipy.spatial.transform import Rotation as R

ROBOT_FRAME = "kinova_base"

# Item ID's (if item == can, slide into grasp pose horizontally)
BOX_ID = 0
CAN_ID = 1

# Item masses
BOX_MASS = 0.3
CAN_MASS = 1.0
SOME_POSE = np.array([0, 0, 0, 0, 0, 0, 1])


def add_paths():
    sys.path.insert(0, "/home/ruic/Documents/opa")
    sys.path.insert(0, "/home/ruic/Documents/opa/opa_comparison/src")


num_objects = 1

HOME_POSE = np.array([0.373, 0.1, 0.13, 0.707, 0.707, 0, 0])
HOME_JOINTS = np.array([-0.705, 0.952, -1.663, -1.927, 2.131, 1.252, -0.438])
POS_DIM = 3
DROP_OFF_OFFSET = np.array([0.0, 0.0, -0.1])

# Only for ablation studies requiring predefined features and
# evaluation of final behavior
DESIRED_ROT_OFFSET = R.from_euler("XYZ", [0, 0, np.pi]).as_quat()

T = 20.0  # taken from FERL yaml file

item_ids = [
    BOX_ID,
    BOX_ID,
    BOX_ID,
]

extra_masses = [
    BOX_MASS,
    BOX_MASS,
    BOX_MASS,
]

# Item pickup poses
start_poses = [
    np.array([0.2, 0.35, -0.08]),
    np.array([0.2, 0.465, -0.08]),
    np.array([0.22, 0.58, -0.08]),
]
start_ori_quats = [
    np.array([0, 1., 0, 0]),
    np.array([0, 1., 0, 0]),
    np.array([0, 1., 0, 0]),
]
start_joints_all = [
    np.array([270.271, 68.32, 221.0, 241.136, 91.637, 36.716, 18.348]),

]

# Item dropoff poses
goal_poses = [
    np.array([0.4, -0.475, 0.1]),
    np.array([0.38, -0.475, 0.1]),
    np.array([0.36, -0.475, 0.1]),
]
goal_ori_quats = start_ori_quats
goal_joints_all = [
    np.array([20, 67.329, 249.304, 289.5, 117.57, 80.329, 92.4]),

]

inspection_poses = [
    np.array([0.7, 0.1, 0.05]),  # sitting at center of table
    np.array([0.6, -0.2, 0.4]),   # standing at corner of table
    np.array([0.6, -0.2, 0.4])   # TODO:
]
inspection_ori_quats = [
    R.from_euler("zyx", [0, 30, 0], degrees=True).as_quat(),
    R.from_euler("zyx", [-20, -20, 0], degrees=True).as_quat(),
    R.from_euler("zyx", [-20, -20, 0], degrees=True).as_quat()  # TODO:
]
