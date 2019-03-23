#!/usr/bin/python3
from solvers import tsr_detect_solver
import pykitti
from utils import constants

solver = tsr_detect_solver.TSRDetectSolver()
solver.run(70)
'''
dataset = pykitti.raw(constants.KITTI_BASE_DIR, constants.DATE, constants.DRIVE)
img_num = 86
solver.visualize(solver.pil_to_cv2(dataset.get_cam2(img_num)), dataset.get_velo(img_num))
'''
