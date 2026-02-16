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
        self._z_hit = 5
        self._z_short = 1
        self._z_max = 1
        self._z_rand = 500
        self._sigma_hit = 30
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
        z_star = float(np.clip(z_star, 0.0, self.laserMax))
        z = float(np.clip(z_reading, 0.0, self.laserMax))
     
        if 0.0 <= z <= self.laserMax:
            pHit = np.exp(-0.5 * (z - z_star) ** 2 / (self._sigma_hit ** 2))
            pHit = pHit / (np.sqrt(2.0 * np.pi) * self._sigma_hit)
        else:
            pHit = 0.0

        if 0.0 <= z <= z_star and self._lambda_short > 1e-12:
            lam = float(self._lambda_short)
            denom = 1.0 - np.exp(-lam * z_star)
            eta = 1.0 / max(denom, 1e-12)
            pShort = eta * lam * np.exp(-lam * z)
        else:
            pShort = 0.0

        eps = self._resolution
        pMax = 1.0 if z >= (self.laserMax - eps) else 0.0

        if 0.0 <= z < self.laserMax:
            pRand = 1.0 / self.laserMax
        else:
            pRand = 0.0

        p = self._z_hit * pHit + self._z_short * pShort + self._z_max * pMax + self._z_rand * pRand
        p /= (self._z_hit + self._z_short + self._z_max + self._z_rand)
        return p, pHit, pShort, pMax, pRand


    def rayCast(self, x_t1):
        beamsRange = np.full(self.nLaser, self.laserMax, dtype=float)
        laserX = np.zeros(self.nLaser, dtype=float)
        laserY = np.zeros(self.nLaser, dtype=float)

        xc = float(x_t1[0])
        yc = float(x_t1[1])
        myPhi = float(x_t1[2])

        L = self._laser_offset
        offSetX = xc + L * np.cos(myPhi)
        offSetY = yc + L * np.sin(myPhi)

        angle_min = -np.pi / 2.0
        angStep = np.pi / float(self.nLaser - 1) if self.nLaser > 1 else 0.0

        r_step = self._resolution
        max_steps = int(self.laserMax / r_step)

        h, w = self._map.shape

        for i in range(self.nLaser):
            ang = self.WrapToPi(myPhi + angle_min + i * angStep)
            c = np.cos(ang)
            s = np.sin(ang)

            for k in range(max_steps + 1):
                rs = k * r_step
                x = offSetX + rs * c
                y = offSetY + rs * s

                xInt = int(np.floor(x / self._resolution))
                yInt = int(np.floor(y / self._resolution))

                if xInt < 0 or xInt >= w or yInt < 0 or yInt >= h:
                    beamsRange[i] = min(rs, self.laserMax)
                    break

                if self._map[yInt, xInt] >= self._min_probability:
                    beamsRange[i] = min(rs, self.laserMax)
                    laserX[i] = xInt
                    laserY[i] = yInt
                    break

        return beamsRange, laserX, laserY


    def beam_range_finder_model(self, z_t1_arr, x_t1):
        z_full = np.asarray(z_t1_arr, dtype=float)
        idx = np.linspace(0, z_full.size - 1, self.nLaser).astype(int)

        z_reading = z_full[idx]
        z_star, _, _ = self.rayCast(x_t1)

        log_q = 0.0
        for i in range(0, self.nLaser, self._subsampling):
            p, _, _, _, _ = self.getProbability(z_star[i], z_reading[i])
            log_q += np.log(max(p, 1e-12))

        return np.exp(log_q)


