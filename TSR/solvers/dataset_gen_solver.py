import bbx_generator
import pykitti
import lidar_image_generator
from utils import constants
import numpy as np
import cv2
import solver
import os

class DatasetGenSolver(solver.Solver):
    def __init__(self, output_base_dir = os.path.join(constants.RESULTS_BASE_DIR, 'Dataset'), threshold = constants.REFLECTIVITY_THRESH, base_dir = constants.KITTI_BASE_DIR, date = constants.DATE, drive = constants.DRIVE):
        solver.Solver.__init__(self, base_dir, date, drive)
        self.__img_gen = lidar_image_generator.LIDARImageGenerator(self._dataset.calib)
        self.__bbx_gen = bbx_generator.BbxGenerator()
        self.__save_dir = output_base_dir

    def get_cropped_imgs(self, img, velo):
        lidar_img = self.__img_gen.generate_refined(velo, img.shape)
        bbxs = self.__bbx_gen.get_bbxs(lidar_img)
        return self.__bbx_gen.crop_to_bbxs(bbxs, img)

    def solve(self, img, velo, base_filename):
        try:
            imgs = self.get_cropped_imgs(img, velo)
            for i, img in enumerate(imgs):
                cv2.imwrite(os.path.join(self.__save_dir, base_filename + '_' + str(i).zfill(2) + '.png'), img)
        except bbx_generator.BbxGenerator.NoClustersFound:
            pass
