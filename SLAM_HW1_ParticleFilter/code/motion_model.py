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
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.01
        self._alpha2 = 0.01
        self._alpha3 = 0.01
        self._alpha4 = 0.01

    def _wrap2pi(self,angle):
        
        return angle - 2*np.pi * np.floor((angle + np.pi) / (2*np.pi))

    def sample(self,mu,sigma):
        return np.random.normal(mu,sigma)

    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """

        # NO MOTION
        if u_t1[0] == u_t0[0] and u_t1[1] == u_t0[1] and u_t1[2] == u_t0[2]:
            return x_t0          

        # MOTION    
        x_t1 = np.zeros_like(x_t0)
        delta_base_rotation = np.arctan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2]
        delta_base_rotation = self._wrap2pi(delta_base_rotation)
        delta_base_translation = np.sqrt((u_t1[0] - u_t0[0])**2 + (u_t1[1] - u_t0[1])**2)
        delta_directional_rotation = u_t1[2] - u_t0[2] - delta_base_rotation
        delta_directional_rotation = self._wrap2pi(delta_directional_rotation)
        
        Rot1 = delta_base_rotation - self.sample(0, self._alpha1 * delta_base_rotation**2 + \
                                self._alpha2 * delta_base_translation**2)
        Trans = delta_base_translation - self.sample(0,self._alpha3 * delta_base_translation**2 + \
                                     self._alpha4 * delta_base_rotation**2 + self._alpha4*delta_directional_rotation**2)
        Rot2 = delta_directional_rotation - self.sample(0, self._alpha1 * delta_directional_rotation**2 + \
                                self._alpha2 * delta_base_translation**2)
        Rot1 = self._wrap2pi(Rot1)
        Rot2 = self._wrap2pi(Rot2)

        x_t1[0] = x_t0[0] + Trans * np.cos(x_t0[2] + Rot1)
        x_t1[1] = x_t0[1] + Trans * np.sin(x_t0[2] + Rot1)
        x_t1[2] = x_t0[2] + Rot1 + Rot2

        return x_t1