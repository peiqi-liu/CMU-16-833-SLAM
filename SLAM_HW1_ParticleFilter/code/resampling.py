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
        M = X_bar.shape[0]

        # Normalize weights
        weights = X_bar[:, 3]
        weights = weights / np.sum(weights)

        # Draw M samples with replacement
        indices = np.random.choice(M, size=M, p=weights)

        # Resample
        X_bar_resampled = X_bar[indices].copy()

        # Reset weights to uniform
        X_bar_resampled[:, 3] = 1.0 / M

        return X_bar_resampled

    def low_variance_sampler(self, X_bar, occupancy_map = None):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        M = X_bar.shape[0]

        weights = X_bar[:, 3]
        weights = weights / np.sum(weights)

        X_bar_resampled = np.zeros_like(X_bar)

        r = np.random.uniform(0, 1.0 / M)
        c = weights[0]
        i = 0

        for m in range(M):
            U = r + m / M
            while U > c:
                i += 1
                c += weights[i]

            X_bar_resampled[m, :] = X_bar[i, :]

        # Reset weights
        X_bar_resampled[:, 3] = 1.0 / M

        return X_bar_resampled