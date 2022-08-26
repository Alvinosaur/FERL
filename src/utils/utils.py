import numpy as np

def normalize_pi_neg_pi(ang):
    while ang > np.pi:
        ang -= 2 * np.pi
    while ang <= -np.pi:
        ang += 2 * np.pi
    return ang