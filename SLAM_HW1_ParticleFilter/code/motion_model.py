'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        # Noise parameters: 
        # alpha1, alpha2: rotation noise
        # alpha3, alpha4: translation noise
        self._alpha1 = 0.001
        self._alpha2 = 0.001
        self._alpha3 = 0.001
        self._alpha4 = 0.001

    def update(self, u_t0, u_t1, x_t0):

        # Odometry motion decomposition
        delta_rot1 = math.atan2(u_t1[1] - u_t0[1],
                                u_t1[0] - u_t0[0]) - u_t0[2]
        delta_trans = math.sqrt((u_t1[0] - u_t0[0])**2 +
                                (u_t1[1] - u_t0[1])**2)
        delta_rot2 = u_t1[2] - u_t0[2] - delta_rot1

        # Add noise
        delta_rot1_hat = delta_rot1 - np.random.normal(
            0,
            math.sqrt(self._alpha1 * delta_rot1**2 +
                    self._alpha2 * delta_trans**2)
        )

        delta_trans_hat = delta_trans - np.random.normal(
            0,
            math.sqrt(self._alpha3 * delta_trans**2 +
                    self._alpha4 * (delta_rot1**2 + delta_rot2**2))
        )

        delta_rot2_hat = delta_rot2 - np.random.normal(
            0,
            math.sqrt(self._alpha1 * delta_rot2**2 +
                    self._alpha2 * delta_trans**2)
        )

        # Update particle pose
        x = x_t0[0] + delta_trans_hat * math.cos(x_t0[2] + delta_rot1_hat)
        y = x_t0[1] + delta_trans_hat * math.sin(x_t0[2] + delta_rot1_hat)
        theta = x_t0[2] + delta_rot1_hat + delta_rot2_hat

        # Normalize angle
        theta = (theta + math.pi) % (2 * math.pi) - math.pi

        return np.array([x, y, theta])