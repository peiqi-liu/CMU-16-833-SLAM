'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
from scipy.stats import norm
import math

class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        # Parameters (tuned for stability)
        self._z_hit = 1
        self._z_short = 0.1
        self._z_max = 0.1
        self._z_rand = 100

        self._sigma_hit = 200.0      # cm
        self._lambda_short = 0.1
        self._max_range = 1000  # cm
        self._min_probability = 0.35
        
        self.occupancy_map = occupancy_map # Expected shape [Height, Width]
        self.resolution = 10 # cm per pixel (standard for these maps)
        self.laser_offset = 25 # cm from robot center to laser
        self.num_beams = 60
        self._subsampling = 180 // self.num_beams  # Use every 10th beam (18 beams total)

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        Calculates the likelihood of a laser scan given a particle pose.
        """

        x, y, theta = x_t1
        prob = 0.0

        for i in range(0, len(z_t1_arr), self._subsampling):

            z = z_t1_arr[i]

            # Ray casting (simplified)
            z_star = self.ray_cast(x, y, theta + np.deg2rad(i - 90))

            # --- p_hit ---
            if 0 <= z <= self._max_range:
                p_hit = norm.pdf(z, z_star, self._sigma_hit)
            else:
                p_hit = 0

            # --- p_short ---
            if 0 <= z <= z_star:
                p_short = self._lambda_short * np.exp(-self._lambda_short * z)
            else:
                p_short = 0

            # --- p_max ---
            p_max = 1.0 if z >= self._max_range else 0.0

            # --- p_rand ---
            if 0 <= z < self._max_range:
                p_rand = 1.0 / self._max_range
            else:
                p_rand = 0

            p = (self._z_hit * p_hit +
                self._z_short * p_short +
                self._z_max * p_max +
                self._z_rand * p_rand)

            prob += np.log(p)

        return np.exp(prob)

    def ray_cast(self, x, y, theta):

        step = 10
        for r in range(0, self._max_range, step):

            x_hit = int(x + r * math.cos(theta))
            y_hit = int(y + r * math.sin(theta))

            if x_hit < 0 or x_hit >= self.occupancy_map.shape[1] \
            or y_hit < 0 or y_hit >= self.occupancy_map.shape[0]:
                return self._max_range

            if self.occupancy_map[y_hit, x_hit] > self._min_probability:
                return r

        return self._max_range