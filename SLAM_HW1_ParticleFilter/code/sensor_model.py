import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.stats import expon
import pdb

class SensorModel:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """

    def __init__(self, occupancy_map):
        """
        Initialize Sensor Model parameters here
        """
        
        self._z_hit = 1000
        self._z_short = 0.01
        self._z_max = 0.03
        self._z_rand = 50000

        self._sigma_hit = 250
        self._lambda_short = 0.01

        self.laser_offset = 25.0

        self.map = occupancy_map

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 8183

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.4

        # Used in sampling angles in ray casting
        self._subsampling = 1

    def get_probability(self, z_tk, x_t, z_tk_star):
        # p_hit
        if 0 <= z_tk <= self._max_range:
            p_hit = (math.exp(-(z_tk - z_tk_star)**2 / (2 * self._sigma_hit**2))) / (
                math.sqrt(2 * math.pi * self._sigma_hit**2)
            )
        else:
            p_hit = 0.0

        # p_short
        if 0 <= z_tk <= z_tk_star:
            eta = 1 / (1 - math.exp(-self._lambda_short * z_tk_star))
            p_short = eta * self._lambda_short * math.exp(-self._lambda_short * z_tk)
        else:
            p_short = 0.0

        # p_max
        if z_tk == self._max_range:
            p_max = 1.0
        else:
            p_max = 0.0

        # p_rand
        if 0 <= z_tk < self._max_range:
            p_rand = 1.0 / self._max_range
        else:
            p_rand = 0.0

        return (
            self._z_hit * p_hit +
            self._z_short * p_short +
            self._z_max * p_max +
            self._z_rand * p_rand
        )

 
    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """

        """
        TODO : Add your code here
        """
        pos_x, pos_y, pos_theta = x_t1
        temp = self.map[min(int(pos_y/10.), 799)][min(int(pos_x/10.), 799)]
        if temp >= self._min_probability or temp == -1:
            return 1e-100
        
        q = 1.0

        # Add 25 cm offset
        laser_x = self.laser_offset* np.cos(pos_theta)
        laser_y = self.laser_offset * np.sin(pos_theta)
        coord_x = int(round((pos_x + laser_x) / 10.0))
        coord_y = int(round((pos_y + laser_y) / 10.0))

        for deg in range(-90, 90, 10 * self._subsampling):
            z_t1_true = self.rayCast(deg, pos_theta, coord_x, coord_y)
            z_t1_k = z_t1_arr[deg+90]
            p = self.get_probability(z_t1_k, x_t1, z_t1_true)
    
            if p > 0:
                q *= p
            else:
                return 1e-50

        return q
  

    def rayCast(self, deg, ang, coord_x, coord_y):
        final_angle= ang + math.radians(deg)
        start_x = coord_x
        start_y = coord_y

        final_x = coord_x
        final_y = coord_y
        while 0 < final_x < self.map.shape[1] and 0 < final_y < self.map.shape[0] and abs(self.map[final_y, final_x]) < 1e-7:
            start_x += 2 * np.cos(final_angle)
            start_y += 2 * np.sin(final_angle)
            final_x = int(round(start_x))
            final_y = int(round(start_y))
        end_p = np.array([final_x,final_y])
        start_p = np.array([coord_x,coord_y])
        dist = np.linalg.norm(end_p-start_p) * 10
        return dist
