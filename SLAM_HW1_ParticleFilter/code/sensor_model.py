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
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self.z_hit = 150
        self.z_short = 17.5
        self.z_max = 15
        self.z_rand = 100
        self.sigma_hit = 100
        self.lambda_short =15
        self.OccMap = occupancy_map
        self.resolution = 10.0
        self.laserMax = 8183.0
        self.nLaser = 60
        self.laser_offset = 25.0

    def _wrap2pi(self, angle):
        return angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))

    def getProbability(self, z_star, z_reading):

        z_star = np.asarray(z_star)
        z_reading = np.asarray(z_reading)

        # ----- HIT -----
        pHit = np.exp(-0.5 * (z_reading - z_star)**2 / self.sigma_hit**2)
        pHit /= (np.sqrt(2 * np.pi) * self.sigma_hit)

        valid_hit = (z_reading >= 0) & (z_reading <= self.laserMax)
        pHit *= valid_hit

        # ----- SHORT -----
        pShort = self.lambda_short * np.exp(-self.lambda_short * z_reading)

        valid_short = (z_reading >= 0) & (z_reading <= z_star)
        pShort *= valid_short

        # ----- MAX -----
        pMax = (z_reading >= self.laserMax).astype(float)

        # ----- RANDOM -----
        pRand = np.ones_like(z_reading) / self.laserMax
        valid_rand = (z_reading >= 0) & (z_reading < self.laserMax)
        pRand *= valid_rand

        # ----- MIXTURE -----
        p = (
            self.z_hit * pHit + \
            self.z_short * pShort + \
            self.z_max * pMax + \
            self.z_rand * pRand \
        )

        p /= (self.z_hit + self.z_short + self.z_max + self.z_rand)

        return p

    def rayCast(self, x_t1):
        # x_t1 is [x, y, theta]
        xc, yc, myPhi = x_t1
        
        # Calculate laser offset position
        # The laser is often slightly ahead of the robot center
        ang_offset = self._wrap2pi(myPhi - np.pi / 2)
        offSetX = xc + self.laser_offset * np.cos(ang_offset)
        offSetY = yc + self.laser_offset * np.sin(ang_offset)

        # 1. Generate all beam angles at once
        angStep = np.pi / self.nLaser
        angles = self._wrap2pi(myPhi - np.pi/2 + angStep * np.arange(1, self.nLaser + 1))

        # 2. Create the range steps
        num_steps = 1000 
        r = np.linspace(0, self.laserMax, num_steps)

        # 3. Use broadcasting to get all (x, y) coordinates for all beams
        # Resulting shapes: (num_steps, nLaser)
        xs = offSetX + r[:, np.newaxis] * np.cos(angles)
        ys = offSetY + r[:, np.newaxis] * np.sin(angles)

        # 4. Map coordinates to occupancy grid indices
        xInt = np.floor(xs / self.resolution).astype(int)
        yInt = np.floor(ys / self.resolution).astype(int)

        # 5. Create a valid mask to prevent index-out-of-bounds
        # Assuming map is 800x800 based on your code
        valid_mask = (xInt >= 0) & (xInt < 800) & (yInt >= 0) & (yInt < 800)

        # 6. Check occupancy
        # We initialize with False and only check valid coordinates
        occ = np.zeros((num_steps, self.nLaser), dtype=bool)
        occ[valid_mask] = np.abs(self.OccMap[yInt[valid_mask], xInt[valid_mask]]) > 0.35
        print(occ.shape)

        # 7. Find the first 'True' in each column (each beam)
        # np.argmax returns the index of the first True. 
        # If no True is found, it returns 0 (which is why we need a hit mask).
        hit_indices = np.argmax(occ, axis=0)
        has_hit = np.any(occ, axis=0)
        print(has_hit.shape)

        # 8. Calculate final ranges
        # Default to laserMax, then update those that actually hit something
        beamsRange = np.full(self.nLaser, self.laserMax)
        beamsRange[has_hit] = r[hit_indices[has_hit]]

        # laserX and laserY are usually for visualization; 
        # if you need them, you can calculate them from beamsRange and angles
        return beamsRange

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """
        # q = 0

        # step = int(180 / self.nLaser)
        # z_reading = [z_t1_arr[n] for n in range(0, 180, step)]
        # zt_star, laserX, laserY = self.rayCast(x_t1)
        # probs = np.zeros(self.nLaser)
        # for i in range(self.nLaser):
        #     probs[i] = self.getProbability(zt_star[i], z_reading[i])
        #     q += np.log(probs[i])

        # q = self.nLaser / np.abs(q)
        # return q

        step = int(180 / self.nLaser)
        z_reading = np.array(z_t1_arr[::step])

        zt_star = self.rayCast(x_t1)

        print(zt_star.shape)

        probs = self.getProbability(zt_star, z_reading)

        q = np.sum(np.log(probs + 1e-12))
        # return q
        return np.exp(q)
