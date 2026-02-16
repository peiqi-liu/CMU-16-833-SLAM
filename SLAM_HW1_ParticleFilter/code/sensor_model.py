'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np

class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        # Parameters (tuned for stability)
        self._z_hit = 10.0
        self._z_short = 0.1
        self._z_max = 0.1
        self._z_rand = 1.0
        
        self._sigma_hit = 150.0
        self._lambda_short = 0.1
        self._max_range = 1000  # cm
        self._min_probability = 0.35
        self._subsampling = 10  # Use every 10th beam (18 beams total)
        
        self.occupancy_map = occupancy_map # Expected shape [Height, Width]
        self.resolution = 10 # cm per pixel (standard for these maps)
        self.laser_offset = 25 # cm from robot center to laser

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        Calculates the likelihood of a laser scan given a particle pose.
        """
        # 1. Subsample laser readings to reduce computation
        z_readings = z_t1_arr[::self._subsampling]
        num_beams = len(z_readings)
        
        # 2. Get expected ranges (z_star) via vectorized ray casting
        z_stars = self._vectorized_ray_cast(x_t1, num_beams)
        
        # 3. Calculate components (Vectorized across all beams)
        # Hit: Gaussian distribution
        p_hit = np.exp(-0.5 * (z_readings - z_stars)**2 / self._sigma_hit**2)
        p_hit /= (self._sigma_hit * np.sqrt(2 * np.pi))
        
        # Short: Exponential distribution
        # Note: eta calculation is simplified; p_short is only for z < z_star
        p_short = np.where(z_readings < z_stars, 
                           self._lambda_short * np.exp(-self._lambda_short * z_readings), 0)
        p_short /= (1 - np.exp(-self._lambda_short * np.maximum(z_stars, 1e-9)))

        # Max: Point mass at max range
        p_max = (z_readings >= self._max_range).astype(float)
        
        # Rand: Uniform distribution
        p_rand = np.where(z_readings < self._max_range, 1.0 / self._max_range, 0)
        
        # 4. Mixture Model
        p_total = (self._z_hit * p_hit + 
                   self._z_short * p_short + 
                   self._z_max * p_max + 
                   self._z_rand * p_rand)
        
        # Normalize weights
        p_total /= (self._z_hit + self._z_short + self._z_max + self._z_rand)
        
        # 5. Log-Likelihood to avoid numerical underflow
        # We sum logs instead of multiplying small decimals
        prob_zt1 = np.exp(np.sum(np.log(p_total + 1e-10)))
        
        return prob_zt1

    def _vectorized_ray_cast(self, x_t1, num_beams):
        """
        Simulates ray casting by projecting multiple distances for all beams at once.
        """
        x, y, theta = x_t1
        
        # Laser position in world frame
        start_x = x + self.laser_offset * np.cos(theta)
        start_y = y + self.laser_offset * np.sin(theta)
        
        # Angles for each beam: from -90 to +90 degrees relative to robot heading
        angles = np.linspace(theta - np.pi/2, theta + np.pi/2, num_beams)
        
        # Define search steps (from 0 to max_range)
        # We step by resolution to ensure we don't jump over walls
        steps = np.arange(0, self._max_range, self.resolution)
        
        # Vectorized projection: [num_steps, num_beams]
        # x_coords = start_x + dist * cos(angle)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        
        # Use broadcasting to get all points at once
        x_pts = start_x + steps[:, np.newaxis] * cos_a
        y_pts = start_y + steps[:, np.newaxis] * sin_a
        
        # Convert to map indices
        ix = (x_pts / self.resolution).astype(int)
        iy = (y_pts / self.resolution).astype(int)
        
        # Check map bounds
        mask = (ix >= 0) & (ix < self.occupancy_map.shape[1]) & \
               (iy >= 0) & (iy < self.occupancy_map.shape[0])
        
        # Initialize z_stars with max range
        z_stars = np.full(num_beams, float(self._max_range))
        
        # For each beam, find the first index where the map is occupied
        for b in range(num_beams):
            beam_mask = mask[:, b]
            # Map lookup: get occupancy values for this specific beam's path
            occupancy_values = self.occupancy_map[iy[beam_mask, b], ix[beam_mask, b]]
            
            # Find first index where occupancy > threshold
            hits = np.where(np.abs(occupancy_values) > self._min_probability)[0]
            if len(hits) > 0:
                z_stars[b] = steps[hits[0]]
                
        return z_stars