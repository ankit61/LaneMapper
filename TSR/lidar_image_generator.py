import velo_projector
import cv2
from utils import constants
import numpy as np

class LIDARImageGenerator(velo_projector.VeloProjector):
    def __init__(self, calib, cam_num = constants.CAM_NUM, min_x = constants.MIN_X):
        velo_projector.VeloProjector.__init__(self, calib, cam_num, min_x)

    def generate_raw(self, velo_points, img_size):
        img_pts_reflectivity = self.project(velo_points)
        lidar_img = np.zeros(img_size[0:2], dtype=np.uint8)
        for (x, y, r) in img_pts_reflectivity:
            try:
                lidar_img[int(y), int(x)] = int(r * 255)
            except IndexError:
                pass

        return lidar_img.astype(np.uint8)

    def generate_refined(self, velo_points, img_size, threshold = constants.REFLECTIVITY_THRESH):
        return self.__refine(self.generate_raw(velo_points, img_size), threshold)

    def __refine(self, lidar_img, threshold = constants.REFLECTIVITY_THRESH):

        _, refined_img = cv2.threshold(lidar_img, threshold, 255, cv2.THRESH_TOZERO)
        refined_img = cv2.dilate(refined_img, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))

        return refined_img
    
