#!/usr/bin/python3
from solvers import bbx_gen_solver
import pykitti
from utils import constants

solver = bbx_gen_solver.BbxGenSolver()
#solver.run(12)

dataset = pykitti.raw(constants.KITTI_BASE_DIR, constants.DATE, constants.DRIVE)
img_num = 86
solver.visualize(solver.pil_to_cv2(dataset.get_cam2(img_num)), dataset.get_velo(img_num))

