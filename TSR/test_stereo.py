#!/usr/bin/python3
import pykitti
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.split(os.path.realpath(__file__))[0], 'utils'))
import constants

basedir = constants.KITTI_BASE_DIR
date    = constants.DATE
drive   = constants.DRIVE

left_cam  = str(constants.CAM_NUM // 2 * 2)
right_cam = str(int(left_cam) + 1)

dataset = pykitti.raw(basedir, date, drive)
gray    = dataset.get_gray(15)

stereo  = cv2.StereoBM_create(64, 9)
disp    = stereo.compute(np.array(gray[0]), np.array(gray[1])).astype(np.float32) / 16.0

calib = dataset.calib

K_L = calib.__getattribute__('K_cam' + left_cam)
K_R = calib.__getattribute__('K_cam' + right_cam)

D_L = calib.__getattribute__('D_cam' + left_cam)
D_R = calib.__getattribute__('D_cam' + right_cam)

img_size = calib.__getattribute__('S_cam' + left_cam)

R_L = calib.__getattribute__('R_cam' + left_cam)
R_R = calib.__getattribute__('R_cam' + right_cam)
R   = np.matmul(R_L.transpose(), R_R)

T_L = calib.__getattribute__('T_cam' + left_cam)
T_R = calib.__getattribute__('T_cam' + right_cam)
T   = T_L - T_R

_, _, _, _, Q, _, _ = cv2.stereoRectify(K_L, D_L, K_R, D_R, (int(img_size[0]), int(img_size[1])), R, T)

points = cv2.reprojectImageTo3D(disp, Q)

f, ax = plt.subplots(1, 1, figsize=(150, 100))
ax.imshow(disp)
ax.set_title('Gray Stereo Disparity')
plt.show()