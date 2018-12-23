import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'utils'))
import constants
import numpy as np
import cv2

class VeloProjector(object):
    
    def __init__(self, calib, cam_num = constants.CAM_NUM, min_x = constants.MIN_X):
        self.__cam_num  = cam_num
        self.__min_x    = min_x
        self.__proj_mat = self.__compute_proj_mat(calib)

    def __prepare_velo_points(self, velo_points):
        processed_velo_points = np.copy(velo_points) #FIXME: can get away with shallow copy, but would be bad programming
        processed_velo_points = processed_velo_points[processed_velo_points[:, 0] > constants.MIN_X, :]
        reflectivity = np.copy(processed_velo_points[:, -1])
        processed_velo_points[:, 3] = 1  #set reflectivity to 1
        return processed_velo_points, reflectivity

    def __compute_proj_mat(self, calib):
        p_rect  = calib.__getattribute__('P_rect_' + str(self.__cam_num) + '0')
        r_rect  = calib.R_rect_00
        tr      = calib.T_cam0_velo_unrect

        return np.matmul(p_rect, np.matmul(r_rect, tr))

    def __project_to_img(self, processed_velo_points):
        dim_norm = self.__proj_mat.shape[0]
        dim_proj = self.__proj_mat.shape[1]

        if(processed_velo_points.shape[1] == dim_proj - 1):
            processed_velo_points = np.concatenate((processed_velo_points, np.ones(processed_velo_points.shape[0], 1)), axis=1)

        assert(processed_velo_points.shape[1] == dim_proj)

        proj_points = np.matmul(self.__proj_mat, processed_velo_points.transpose()).transpose()
        #proj_points[:, :-1] = proj_points[:, :-1] / proj_points[:, -1][:, None]

        for i in range(proj_points.shape[0]):
            for j in range(proj_points.shape[1] - 1):
                proj_points[i, j] /= proj_points[i, -1]

        return proj_points[:,:-1]

    def project(self, velo_points):
        velo_points, reflectivity  = self.__prepare_velo_points(velo_points)
        proj_points  = self.__project_to_img(velo_points)

        img_pts_reflectivity = np.zeros((proj_points.shape[0], proj_points.shape[1] + 1))
        img_pts_reflectivity[:, -1]  = reflectivity
        img_pts_reflectivity[:, :-1] = proj_points
        
        return img_pts_reflectivity

    def generate_viz_img(self, img_pts_reflectivity, img):
        viz_img = img.copy()
        for (x, y, r) in img_pts_reflectivity:
            cv2.circle(viz_img, (int(x), int(y)), 5, (int(r * 255), 0, 0))  #draw blue circles
        return viz_img