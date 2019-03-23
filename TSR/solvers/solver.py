import pykitti
import cv2
import numpy as np
import abc
from utils import constants
from tqdm import tqdm
from PIL import Image

class Solver(abc.ABC):

    class NotVisualizable(NotImplementedError): pass

    def __init__(self, base_dir = constants.KITTI_BASE_DIR, date = constants.DATE, drive = constants.DRIVE, cam_num = constants.CAM_NUM):
        self._dataset = pykitti.raw(base_dir, date, drive)
        self.__get_cam = self._dataset.__getattribute__('get_cam' + str(cam_num))

    def pil_to_cv2(self, pil_img):
        img = np.array(pil_img)
        if(img.shape[2] == 3):
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif(img.shape[2] == 1):
            return np.array(img)

    def cv2_to_pil(self, cv2_img):
        if(cv2_img.shape[2] == 3):
            return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
        elif(cv2_img.shape[2] == 1):
            return Image.fromarray(cv2_img)

    def run(self, start = 0, end = 10000):
        for i in tqdm(range(start, min(end, len(self._dataset)))):
            self.solve(self.pil_to_cv2(self.__get_cam(i)), self._dataset.get_velo(i), self._dataset.get_base_filename(i))

    def show_result_for(self, idx):
        self.visualize(self.pil_to_cv2(self.__get_cam(idx)), self._dataset.get_velo(idx))

    @abc.abstractmethod
    def solve(self, img, velo, base_filename):
        pass

    def visualize(self, img, velo):
        raise NotVisualizable('this can\'t be visualized!')