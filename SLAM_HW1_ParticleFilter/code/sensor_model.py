'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


class SensorModel:
    def __init__(self, occupancy_map):
        self._z_hit = 100
        self._z_short = 1
        self._z_max = 1
        self._z_rand = 5000
        self._sigma_hit = 100
        self._lambda_short = 10
        self._min_probability = 0.35

        self._map = occupancy_map
        self._resolution = 10.0

        self.laserMax = 8000.0
        self.nLaser = 90
        self._subsampling = 180 // self.nLaser
        self._laser_offset = 25.0

    def WrapToPi(self, angle):
        return angle - 2.0 * np.pi * np.floor((angle + np.pi) / (2.0 * np.pi))

    def getProbability(self, z_star, z_reading):
        z_star = np.clip(z_star, 0.0, self.laserMax).astype(float)
        z = np.clip(z_reading, 0.0, self.laserMax).astype(float)

        # --- pHit ---
        pHit = np.exp(-0.5 * (z - z_star) ** 2 / (self._sigma_hit ** 2))
        pHit /= (np.sqrt(2.0 * np.pi) * self._sigma_hit)

        # --- pShort ---
        lam = self._lambda_short
        pShort = np.zeros_like(z)

        valid_short = (z >= 0.0) & (z <= z_star) & (lam > 1e-12)
        if lam > 1e-12:
            denom = 1.0 - np.exp(-lam * z_star)
            eta = 1.0 / np.maximum(denom, 1e-12)
            pShort[valid_short] = eta[valid_short] * lam * np.exp(-lam * z[valid_short])

        # --- pMax ---
        eps = self._resolution
        pMax = (z >= (self.laserMax - eps)).astype(float)

        # --- pRand ---
        pRand = np.zeros_like(z)
        valid_rand = (z >= 0.0) & (z < self.laserMax)
        pRand[valid_rand] = 1.0 / self.laserMax

        # --- Mixture ---
        p = (
            self._z_hit * pHit +
            self._z_short * pShort +
            self._z_max * pMax +
            self._z_rand * pRand
        )

        p /= (self._z_hit + self._z_short + self._z_max + self._z_rand)

        return p, pHit, pShort, pMax, pRand

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        z_full = np.asarray(z_t1_arr, dtype=float)
        idx = np.linspace(0, z_full.size - 1, self.nLaser).astype(int)

        z_reading = z_full[idx]
        z_star, _, _ = self.rayCast(x_t1)

        # Subsample
        mask = np.zeros(self.nLaser, dtype=bool)
        mask[::self._subsampling] = True

        z_star_sub = z_star[mask]
        z_read_sub = z_reading[mask]

        p, _, _, _, _ = self.getProbability(z_star_sub, z_read_sub)

        log_q = np.sum(np.log(np.maximum(p, 1e-12)))

        return np.exp(log_q)


    def rayCast(self, x_t1):
        beamsRange = np.full(self.nLaser, self.laserMax, dtype=float)
        laserX = np.zeros(self.nLaser, dtype=float)
        laserY = np.zeros(self.nLaser, dtype=float)

        xc = float(x_t1[0])
        yc = float(x_t1[1])
        myPhi = float(x_t1[2])

        # Laser offset
        L = self._laser_offset
        offSetX = xc + L * np.cos(myPhi)
        offSetY = yc + L * np.sin(myPhi)

        # Beam angles
        angle_min = -np.pi / 2.0
        if self.nLaser > 1:
            angStep = np.pi / float(self.nLaser - 1)
        else:
            angStep = 0.0

        angles = myPhi + angle_min + np.arange(self.nLaser) * angStep
        angles = np.arctan2(np.sin(angles), np.cos(angles))  # WrapToPi

        cos_vals = np.cos(angles)
        sin_vals = np.sin(angles)

        # Precompute ray steps
        r_step = self._resolution
        max_steps = int(self.laserMax / r_step)
        rs = np.arange(max_steps + 1) * r_step  # shape: (K,)

        h, w = self._map.shape

        for i in range(self.nLaser):

            # All ray points at once
            x = offSetX + rs * cos_vals[i]
            y = offSetY + rs * sin_vals[i]

            xInt = np.floor(x / self._resolution).astype(int)
            yInt = np.floor(y / self._resolution).astype(int)

            # Valid map indices
            valid = (
                (xInt >= 0) & (xInt < w) &
                (yInt >= 0) & (yInt < h)
            )

            # Stop if out of bounds
            if not np.any(valid):
                continue

            occ = np.zeros_like(valid)
            occ[valid] = (
                self._map[yInt[valid], xInt[valid]] >= self._min_probability
            )

            # Combine hit or out-of-bounds
            hit_mask = occ | (~valid)

            if np.any(hit_mask):
                hit_idx = np.argmax(hit_mask)
                beamsRange[i] = min(rs[hit_idx], self.laserMax)

                if valid[hit_idx]:
                    laserX[i] = xInt[hit_idx]
                    laserY[i] = yInt[hit_idx]

        return beamsRange, laserX, laserY
