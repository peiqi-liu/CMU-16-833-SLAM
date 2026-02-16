'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np


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
    

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        M = X_bar.shape[0]
        X_bar_resampled = np.zeros_like(X_bar)

        # Normalize weights (do NOT modify X_bar in-place)
        weights = X_bar[:, 3].astype(float)
        w_sum = np.sum(weights)
        if w_sum <= 0.0 or not np.isfinite(w_sum):
            weights = np.ones(M, dtype=float) / M
        else:
            weights = weights / w_sum

        r = np.random.uniform(0.0, 1.0 / M)
        c = weights[0]
        i = 0

        for m in range(M):
            U = r + m / M
            while U > c:
                i += 1
                # guard (numerical issues)
                if i >= M:
                    i = M - 1
                    break
                c += weights[i]

            X_bar_resampled[m, :] = X_bar[i, :]

        # Reset weights to uniform after resampling (standard PF)
        X_bar_resampled[:, 3] = 1.0 / M

        return X_bar_resampled


