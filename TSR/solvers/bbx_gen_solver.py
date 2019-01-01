#!/usr/bin/python3
import bbx_generator
import pykitti
import lidar_image_generator
from utils import constants
import numpy as np
import cv2
import solver
import os
import lidar_image_generator

class BbxGenSolver(solver.Solver):
    def __init__(self, output_base_dir = 'Results/TSR_BBx', threshold = constants.REFLECTIVITY_THRESH, base_dir = constants.KITTI_BASE_DIR, date = constants.DATE, drive = constants.DRIVE):
        solver.Solver.__init__(self, base_dir, date, drive)
        self.__img_gen = lidar_image_generator.LIDARImageGenerator(self._dataset.calib)
        self.__bbx_gen = bbx_generator.BbxGenerator()
        self.__save_dir = os.path.join(constants.BASE_DIR, output_base_dir)

    def get_bbx_img(self, img, velo):
        lidar_img = self.__img_gen.generate_refined(velo, img.shape)
        bbxs = self.__bbx_gen.get_bbxs(lidar_img)
        return self.__bbx_gen.draw_bbxs(bbxs, img)

    def solve(self, img, velo, base_filename):
        try:
            bbx_img = self.get_bbx_img(img, velo)
            cv2.imwrite(os.path.join(self.__save_dir, base_filename + '.png'), bbx_img)
        except bbx_generator.BbxGenerator.NoClustersFound:
            pass

    def visualize(self, img, velo):
        bbx_img = self.get_bbx_img(img, velo)
        cv2.imshow('debug', bbx_img)
        cv2.waitKey(0)