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
        # 1. Calculate the relative motion in the odometry frame
        # We find the 'delta' between the two odometry readings
        dx = u_t1[0] - u_t0[0]
        dy = u_t1[1] - u_t0[1]
        
        # Initial rotation to point toward the new position
        delta_rot1 = np.arctan2(dy, dx) - u_t0[2]
        # Straight line distance
        delta_trans = np.sqrt(dx**2 + dy**2)
        # Final rotation to match the current heading
        delta_rot2 = u_t1[2] - u_t0[2] - delta_rot1

        # Handle the case where the robot didn't move much (atan2 becomes unstable)
        if delta_trans < 1e-3:
            delta_rot1 = 0.0
            delta_rot2 = u_t1[2] - u_t0[2]

        # 2. Sample the noisy movement
        # Each particle gets its own unique noise based on the alphas
        num_particles = 1 if x_t0.ndim == 1 else x_t0.shape[0]
        
        # Calculate variances for the noise
        var_rot1 = self._alpha1 * delta_rot1**2 + self._alpha2 * delta_trans**2
        var_trans = self._alpha3 * delta_trans**2 + self._alpha4 * (delta_rot1**2 + delta_rot2**2)
        var_rot2 = self._alpha1 * delta_rot2**2 + self._alpha2 * delta_trans**2

        # Generate noisy deltas (Normal distribution)
        noisy_rot1 = delta_rot1 - np.random.normal(0, np.sqrt(np.abs(var_rot1)), num_particles)
        noisy_trans = delta_trans - np.random.normal(0, np.sqrt(np.abs(var_trans)), num_particles)
        noisy_rot2 = delta_rot2 - np.random.normal(0, np.sqrt(np.abs(var_rot2)), num_particles)

        # 3. Apply the noisy deltas to the particle's previous world state
        # x_t1 = [x, y, theta]
        x_old = x_t0[..., 0]
        y_old = x_t0[..., 1]
        theta_old = x_t0[..., 2]

        x_new = x_old + noisy_trans * np.cos(theta_old + noisy_rot1)
        y_new = y_old + noisy_trans * np.sin(theta_old + noisy_rot1)
        theta_new = self._wrap2pi(theta_old + noisy_rot1 + noisy_rot2)

        # Combine back into [N, 3] array
        x_t1 = np.hstack((x_new, y_new, theta_new))
        
        return x_t1