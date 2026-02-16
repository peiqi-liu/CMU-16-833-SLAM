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

        # Normalize weights
        # log_weights = X_bar[:, 3]
        # max_log_weight = np.max(log_weights)  # For numerical stability
        # weights = np.exp(log_weights - max_log_weight)
        # weights /= np.sum(weights + 1e-15)  # Normalize

        weights = X_bar[:, 3]
        weights = weights / np.sum(weights)

        # Cumulative sum
        cumulative_sum = np.cumsum(weights)

        # Systematic samples
        r = np.random.uniform(0, 1.0 / M)
        u = r + np.arange(M) / M

        # Find indices
        indices = np.searchsorted(cumulative_sum, u)

        # Resample
        X_bar_resampled = X_bar[indices].copy()

        # Reset weights to uniform
        X_bar_resampled[:, 3] = 1.0 / M

        # -------------------------------------------------
        # Kidnapped robot recovery
        # -------------------------------------------------
        if occupancy_map is not None:
            from main import init_particles_freespace
            alpha = 0.005
            n_random = int(alpha * M)

            if n_random > 0:
                random_indices = np.random.choice(M, n_random, replace=False)

                X_bar_resampled[random_indices, :] = init_particles_freespace(n_random, occupancy_map)
                X_bar_resampled[random_indices, 3] = 1.0 / M

        return X_bar_resampled