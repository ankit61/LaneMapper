#!/usr/bin/python3

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys

def plot_lanes(left_lane_file, right_lane_file, ax):
    left_fin    = open(left_lane_file)
    right_fin   = open(right_lane_file)

    left_pts = np.array([np.array(line.split()).astype(np.float) for line in left_fin if line.strip() != ''])
    right_pts = np.array([np.array(line.split()).astype(np.float) for line in right_fin if line.strip() != ''])

    ax.scatter(left_pts[:, 0], left_pts[:, 1], left_pts[:, 2], c='r')
    ax.scatter(right_pts[:, 0], right_pts[:, 1], right_pts[:, 2], c='b')

    ax.set_xlabel('Forward')
    ax.set_ylabel('Leftward')
    ax.set_zlabel('Upward')

    left_fin.close()
    right_fin.close()

def plot_traffic_signs(vio_file, traffic_signs_file, ax):
    tsr_fin = open(traffic_signs_file)
    vio_data = np.array([np.array(line.split())[[11, 3, 7]].astype(np.float) for line in open(vio_file)])
    
    for line in tsr_fin:
        num, signs = line.split(':', 1)
        num = int(num)
        for sign in signs.split('\t'):
            sign = sign.strip()
            ax.scatter(vio_data[num, 0], vio_data[num, 1], vio_data[num, 2], marker='*' , label = sign[1:].split(',')[0])

if(__name__ == '__main__'):
    left_lane_file      = 'Results/2011_09_29_0004/new_left.txt'
    right_lane_file     = 'Results/2011_09_29_0004/new_right.txt'
    vio_file            = 'Datasets/KITTI/2011_09_29/2011_09_29_drive_0004_sync/vio_0004.txt'
    traffic_signs_file  = 'TSR/traffic_signs.txt'
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_lanes(left_lane_file, right_lane_file, ax)
    plot_traffic_signs(vio_file, traffic_signs_file, ax)
    ax.legend()
    plt.show()
