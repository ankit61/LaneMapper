#!/usr/bin/python3
import pykitti
import velo_projector
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'utils'))
import constants
import cv2
import numpy as np

dataset = pykitti.raw(constants.KITTI_BASE_DIR , constants.DATE, constants.DRIVE)
test_img = 0

vp = velo_projector.VeloProjector(dataset.calib)
img = cv2.cvtColor(np.array(dataset.get_rgb(0)[0]), cv2.COLOR_RGB2BGR)
vp(dataset.get_velo(test_img), True, img)
