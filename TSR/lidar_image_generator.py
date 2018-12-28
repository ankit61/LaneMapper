import velo_projector
import cv2
from utils import constants
from utils import LIDARFactor
import numpy as np

class LIDARImageGenerator(velo_projector.VeloProjector):
    def __init__(self, calib, cam_num = constants.CAM_NUM, min_x = constants.MIN_X):
        velo_projector.VeloProjector.__init__(self, calib, cam_num, min_x)

    def generate(self, velo_points, img_size, intensity_type = LIDARFactor.REFLECTIVITY):
        img_pts_reflectivity = self.project(velo_points)
        img = np.zeros(img_size, dtype=np.uint8)
        for (x, y, r, d) in img_pts_reflectivity:
            try:
                img[int(y), int(x)] = int(r * 255) if intensity_type == LIDARFactor.REFLECTIVITY else int(d * 255)
            except IndexError:
                pass

        return img