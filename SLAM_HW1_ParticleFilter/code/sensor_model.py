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
        self._z_hit = 150
        self._z_short = 2
        self._z_max = 0.5
        self._z_rand = 100

        self._sigma_hit = 50.0      # cm
        self._lambda_short = 0.01
        self._max_range = 1000  # cm
        self._min_probability = 0.35
        
        self.occupancy_map = occupancy_map # Expected shape [Height, Width]
        self.resolution = 10 # cm per pixel (standard for these maps)
        self.laser_offset = 25 # cm from robot center to laser
        self.num_beams = 30
        self._subsampling = 180 // self.num_beams  # Use every 10th beam (18 beams total)

    def visualize_scan(self, z_t1_arr, x_t1):
        """
        Visualizes the expected rays vs actual sensor readings.
        """
        import matplotlib.pyplot as plt

        # 1. Setup Data
        z_readings = z_t1_arr[::self._subsampling]
        num_beams = len(z_readings)
        z_stars = self._vectorized_ray_cast(x_t1, num_beams)
        
        x, y, theta = x_t1
        angles = np.linspace(theta - np.pi/2, theta + np.pi/2, num_beams)
        
        # 2. Convert Polar (range, angle) to World Coordinates (x, y)
        # Laser start position
        ls_x = x + self.laser_offset * np.cos(theta)
        ls_y = y + self.laser_offset * np.sin(theta)
        
        # Actual readings endpoints
        act_x = ls_x + z_readings * np.cos(angles)
        act_y = ls_y + z_readings * np.sin(angles)
        
        # Expected hits endpoints
        exp_x = ls_x + z_stars * np.cos(angles)
        exp_y = ls_y + z_stars * np.sin(angles)

        # 3. Plotting
        plt.figure(figsize=(4, 4))
        # Note: Use [row, col] -> [y, x] logic for map display
        plt.imshow(self.occupancy_map, cmap='Greys', origin='lower', 
                extent=[0, self.occupancy_map.shape[1] * self.resolution, 
                        0, self.occupancy_map.shape[0] * self.resolution])
        
        # Plot robot pose
        plt.arrow(x, y, 20 * np.cos(theta), 20 * np.sin(theta), head_width=10, color='red', label='Robot Pose')
        
        # Plot laser starting point
        plt.plot(ls_x, ls_y, 'ro', markersize=5)

        # Plot Actual Readings (Blue) vs Expected Hits (Green)
        plt.scatter(act_x, act_y, s=10, c='blue', label='Actual Scan (z)')
        plt.scatter(exp_x, exp_y, s=10, c='green', label='Ray Cast (z*)')
        
        # Draw lines for the first and last beam to check FOV
        plt.plot([ls_x, act_x[0]], [ls_y, act_y[0]], 'b--', alpha=0.3)
        plt.plot([ls_x, act_x[-1]], [ls_y, act_y[-1]], 'b--', alpha=0.3)

        plt.title(f"Likelihood Check | Pose: {np.round(x_t1, 2)}")
        plt.legend()
        plt.show()
        plt.pause(0.00001)

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        Calculates the likelihood of a laser scan given a particle pose.
        """
        # self.visualize_scan(z_t1_arr, x_t1)
        # 1. Subsample laser readings to reduce computation
        z_readings = z_t1_arr[::self._subsampling]
        
        # 2. Get expected ranges (z_star) via vectorized ray casting
        z_stars = self._vectorized_ray_cast(x_t1, self.num_beams)
        
        # 3. Calculate components (Vectorized across all beams)
        p_hit = np.exp(-0.5 * (z_readings - z_stars)**2 / self._sigma_hit**2)
        p_hit /= (self._sigma_hit * np.sqrt(2 * np.pi))
        
        # Short: Exponential distribution
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
        # prob_zt1 = np.exp(np.sum(np.log(p_total + 1e-10)))
        log_sum = np.sum(np.log(p_total + 1e-10))
        return log_sum / np.abs(log_sum)

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
        steps = np.arange(0, self._max_range, self.resolution / 2)
        
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