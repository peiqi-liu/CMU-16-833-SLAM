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
        self.map = occupancy_map
        self.Z_MAX = 8183
        self.P_HIT_SIGMA = 250
        self.P_SHORT_LAMBDA = 0.01
        self.Z_PHIT = 1000
        self.Z_PSHORT = 0.01
        self.Z_PMAX = 0.03
        self.Z_PRAND = 50000

    def p_hit(self, z_tk, x_t, z_tk_star):
        if 0 <= z_tk <= self.Z_MAX:
            gaussian = (math.exp(-(z_tk - z_tk_star)**2 / (2 * self.P_HIT_SIGMA**2)))/ math.sqrt(2 * math.pi * self.P_HIT_SIGMA**2)
            return gaussian
        else:
            return 0.0

    def p_short(self, z_tk, x_t, z_tk_star):
        if 0 <= z_tk <= z_tk_star:
            eta = 1 / (1 - math.exp(-self.P_SHORT_LAMBDA * z_tk_star))
            return eta * self.P_SHORT_LAMBDA * math.exp(-self.P_SHORT_LAMBDA * z_tk)
        else:
            return 0.0

    def p_max(self, z_tk, x_t):
        if z_tk == self.Z_MAX:
            return 1.0
        else:
            return 0.0

    def p_rand(self, z_tk, x_t):
        if 0 <= z_tk < self.Z_MAX:
            return 1.0 / self.Z_MAX
        else:
            return 0.0

 
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
        if temp > 0.4 or temp == -1:
            return 1e-100
        q = 0.0

        laser_x = 25.0 * np.cos(pos_theta)
        laser_y = 25.0 * np.sin(pos_theta)
        coord_x = int(round((pos_x + laser_x) / 10.0))
        coord_y = int(round((pos_y + laser_y) / 10.0))

        for deg in range (-90,90, 10):
            z_t1_true = self.rayCast(deg, pos_theta, coord_x, coord_y)
            z_t1_k = z_t1_arr[deg+90]
            p1 = self.Z_PHIT * self.p_hit(z_t1_k, x_t1, z_t1_true)
            p2 = self.Z_PSHORT * self.p_short(z_t1_k, x_t1, z_t1_true)
            p3 = self.Z_PMAX * self.p_max(z_t1_k, x_t1)
            p4 = self.Z_PRAND * self.p_rand(z_t1_k, x_t1)
            p = p1 + p2 + p3 + p4
            # p /= (p1 + p2 + p3 + p4)
            if p > 0:
                q = q + np.log(p)
        return math.exp(q)


    def rayCast(self, deg, ang, coord_x, coord_y):
        final_angle= ang + math.radians(deg)
        start_x = coord_x
        start_y = coord_y
        final_x = coord_x
        final_y = coord_y
        while 0 < final_x < self.map.shape[1] and 0 < final_y < self.map.shape[0] and abs(self.map[final_y, final_x]) < 0.0000001:
            start_x += 2 * np.cos(final_angle)
            start_y += 2 * np.sin(final_angle)
            final_x = int(round(start_x))
            final_y = int(round(start_y))
        end_p = np.array([final_x,final_y])
        start_p = np.array([coord_x,coord_y])
        dist = np.linalg.norm(end_p-start_p) * 10
        return dist