'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import argparse
import numpy as np
import sys, os

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling

from matplotlib import pyplot as plt

def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])


def visualize_timestep(X_bar, tstep, output_path):
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o')
    plt.savefig('{}/{:04d}.png'.format(output_path, tstep))
    plt.pause(0.00001)
    scat.remove()


def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


def init_particles_freespace(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles

    """
    TODO : Add your code here
    """ 
    x, y = np.where(occupancy_map == 0)
    idx = np.random.choice(np.arange(len(x)), num_particles, replace=False)
    x0_vals = y[idx].reshape(len(idx),1) * 10.
    y0_vals = x[idx].reshape(len(idx),1) * 10.
    theta0_vals = np.random.uniform( -3.14, 3.14, (num_particles, 1) )
    w0_vals = np.ones( (num_particles,1), dtype=np.float64)
    w0_vals = w0_vals / num_particles
    X_bar_init = np.hstack((x0_vals,y0_vals,theta0_vals,w0_vals))
    return X_bar_init

if __name__ == '__main__':
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    """
    Initialize Parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata1.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=1500, type=int)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', default=11203, type=int)
    parser.add_argument('--vectorized', action='store_true',
                    help='Use vectorized motion + vectorized resampling (bonus)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    num_particles = args.num_particles
    # X_bar = init_particles_random(num_particles, occupancy_map)
    X_bar = init_particles_freespace(num_particles, occupancy_map)
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if args.visualize:
        visualize_map(occupancy_map)

    first_time_idx = True
    for time_idx, line in enumerate(logfile):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        meas_type = line[0] # L : laser scan measurement, O : odometry measurement
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ') # convert measurement values from string to double

        odometry_robot = meas_vals[0:3] # odometry reading [x, y, theta] in odometry frame
        time_stamp = meas_vals[-1]

        # ignore pure odometry measurements for (faster debugging)
        # if ((time_stamp <= 0.0) | (meas_type == "O")):
        #     continue

        if (meas_type == "L"):
             odometry_laser = meas_vals[3:6] # [x, y, theta] coordinates of laser in odometry frame
             ranges = meas_vals[6:-1] # 180 range measurement values from single laser scan
        
        print("Processing time step " + str(time_idx) + " at time " + str(time_stamp) + "s")

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot

        # Note: this formulation is intuitive but not vectorized; looping in python is SLOW.
        # Vectorized version will receive a bonus. i.e., the functions take all particles as the input and process them in a vector.
        # if meas_type == "L":
        #     x_mean = X_bar[:, 0:3].mean(axis=0)
        #     sensor_model.visualize_scan(ranges, x_mean)
        # Note: this formulation is intuitive but not vectorized; looping in python is SLOW.

        if args.vectorized:
            X_pred = motion_model.update_vectorization(u_t0, u_t1, X_bar[:, 0:3])  # (M,3)
            X_bar_new[:, 0:3] = X_pred

            if meas_type == "L":
                z_t = ranges
                w = np.empty(num_particles, dtype=np.float64)
                for m in range(num_particles):
                    w[m] = sensor_model.beam_range_finder_model(z_t, X_pred[m])
                X_bar_new[:, 3] = w
            else:
                X_bar_new[:, 3] = X_bar[:, 3]

        else:

            for m in range(num_particles):

                # MOTION MODEL
                x_t0 = X_bar[m, 0:3]
                x_t1 = motion_model.update(u_t0, u_t1, x_t0)

                # SENSOR MODEL
                if meas_type == "L":
                    z_t = ranges
                    w_t = sensor_model.beam_range_finder_model(z_t, x_t1)
                    X_bar_new[m, :] = np.hstack((x_t1, w_t))
                else:
                    X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))

        
        X_bar = X_bar_new
        u_t0 = u_t1

        """
        RESAMPLING
        """
   
        X_bar = resampler.low_variance_sampler(X_bar, vectorized=args.vectorized)
        
        
        # if meas_type == "L":
        #     if time_idx % 10 == 0:
        #         X_bar = resampler.low_variance_sampler(X_bar, occupancy_map)
        #     else:
        #         X_bar = resampler.low_variance_sampler(X_bar)

        if args.visualize:
            visualize_timestep(X_bar, time_idx, args.output)