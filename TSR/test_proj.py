#!/usr/bin/python3
import pykitti
import lidar_image_generator
import sys, os
from utils import constants
import cv2
import numpy as np
from utils import LIDARFactor

dataset = pykitti.raw(constants.KITTI_BASE_DIR , constants.DATE, constants.DRIVE)
test_img_num = 36
threshold = 0

vp = lidar_image_generator.LIDARImageGenerator(dataset.calib)
img = cv2.cvtColor(np.array(dataset.get_rgb(test_img_num)[0]), cv2.COLOR_RGB2BGR)
img = img.astype(np.uint8)

viz_img = vp.generate(dataset.get_velo(test_img_num), img.shape)
_, viz_img = cv2.threshold(viz_img, threshold, 255, cv2.THRESH_TOZERO)
viz_img = cv2.dilate(viz_img, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))
viz_img = cv2.applyColorMap(viz_img, cv2.COLORMAP_JET)
cv2.imshow('debug', viz_img)
cv2.waitKey(0)