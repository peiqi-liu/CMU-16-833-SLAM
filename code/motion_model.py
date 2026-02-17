import numpy as np
import math

class MotionModel:
    """
    Odometry motion model (Thrun, Burgard, Fox - Probabilistic Robotics, Ch. 5).
    """
    def __init__(self):
        # These MUST be tuned for your dataset
        self._alpha1 = 0.0001  # rot noise from rot
        self._alpha2 = 0  # rot noise from trans
        self._alpha3 = 0.01  # trans noise from trans
        self._alpha4 = 0  # trans noise from rot

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _sample_normal(std):
        if std <= 0.0:
            return 0.0
        return np.random.normal(0.0, std)

    def update(self, u_t0, u_t1, x_t0):
        """
        u_t0: [x, y, theta] odom at t-1   (odom frame)
        u_t1: [x, y, theta] odom at t     (odom frame)
        x_t0: [x, y, theta] particle pose at t-1 (world frame)

        returns x_t1: particle pose at t (world frame)
        """
        x0_odom, y0_odom, th0_odom = u_t0
        x1_odom, y1_odom, th1_odom = u_t1

        # # check if no motion
        # if (x1_odom == x0_odom) and (y1_odom == y0_odom) and (th1_odom == th0_odom):
        #     return x_t0

        dx = x1_odom - x0_odom
        dy = y1_odom - y0_odom


        deltaR1 = np.atan2(dy, dx) - th0_odom
        deltaTrans = np.hypot(dx, dy)
        deltaR2 = th1_odom - th0_odom - deltaR1

        rot1_var = self._alpha1 * (deltaR1 ** 2) + self._alpha2 * (deltaTrans ** 2)
        trans_var = (self._alpha3 * (deltaTrans ** 2) + self._alpha4 * (deltaR1 ** 2) + self._alpha4 * (deltaR2 ** 2))
        rot2_var = self._alpha1 * (deltaR2 ** 2) + self._alpha2 * (deltaTrans ** 2)

        rot1_hat = deltaR1 - self._sample_normal(np.sqrt(rot1_var))
        trans_hat = deltaTrans - self._sample_normal(np.sqrt(trans_var))
        rot2_hat = deltaR2 - self._sample_normal(np.sqrt(rot2_var))

        x, y, th = x_t0
        x_new = x + trans_hat * np.cos(th + rot1_hat)
        y_new = y + trans_hat * np.sin(th + rot1_hat)
        th_new = th + rot1_hat + rot2_hat
        th_new = self._wrap_to_pi(th + rot1_hat + rot2_hat)

        return np.array([x_new, y_new, th_new])


