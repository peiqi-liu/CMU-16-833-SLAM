'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import pdb
import random
class Resampling:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """

    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled =  np.zeros_like(X_bar)
        return X_bar_resampled
    
    
    def low_variance_sampler(self, X_bar, occupancy_map=None):
        """
        param[in]  X_bar : [M x 4] array [x, y, theta, wt]
        param[out] X_bar_resampled : [M x 4] resampled particles
        """

        M = len(X_bar)
        X_bar_resampled = []

        # ---- Normalize weights safely ----
        wt = X_bar[:, 3].astype(np.float64)
        w_sum = wt.sum()

        if w_sum <= 0 or not np.isfinite(w_sum):
            wt[:] = 1.0 / M
        else:
            wt /= w_sum

        # ---- Low variance resampling ----
        r = random.uniform(0, 1.0 / M)
        c = wt[0]
        i = 0

        for m in range(M):
            u = r + m * (1.0 / M)
            while u > c:
                i += 1
                c += wt[i]
            X_bar_resampled.append(X_bar[i].copy())

        X_bar_resampled = np.asarray(X_bar_resampled)

        # ---- Robot Kidnapping Recovery ----
        if occupancy_map is not None:
            M_rand = 5

            if M_rand > 0:
                from main import init_particles_freespace

                X_random = init_particles_freespace(
                    num_particles=M_rand,
                    occupancy_map=occupancy_map
                )

                # Replace random subset of particles
                replace_idx = np.random.choice(M, M_rand, replace=False)
                X_bar_resampled[replace_idx] = X_random

        # ---- Reset weights uniformly ----
        X_bar_resampled[:, 3] = 1.0 / M

        return X_bar_resampled
    

    def low_variance_sampler_vectorized(self, X_bar):
        M = X_bar.shape[0]

        w = X_bar[:, 3].astype(np.float64)
        w_sum = w.sum()
        if w_sum <= 0 or not np.isfinite(w_sum):
            w[:] = 1.0 / M
        else:
            w /= w_sum

        cdf = np.cumsum(w)
        cdf[-1] = 1.0  

        r = np.random.uniform(0.0, 1.0 / M)
        u = r + (np.arange(M) / M)

        idx = np.searchsorted(cdf, u, side="left")
        X_resampled = X_bar[idx].copy()

        X_resampled[:, 3] = 1.0 / M
        return X_resampled




if __name__ == "__main__":
    pass