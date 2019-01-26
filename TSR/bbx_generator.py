from sklearn import cluster
from utils import constants
import math
from scipy.spatial import distance
import cv2
import numpy as np
from python_algorithms.basic import union_find
import math

class BbxGenerator():

    class NoClustersFound(Exception):
        pass

    def __init__(self, eps = constants.EPS, min_samples = constants.MIN_PTS, min_intensity_diff = constants.MIN_INTENSITY_DIFF):
        self.__eps = eps
        self.__min_samples = min_samples
        self.__min_intensity_diff = min_intensity_diff

    def __dist(self, p1, p2):
        '''
            p1, p2: [x, y, r]
        '''

        if(abs(p1[2] - p2[2]) > self.__min_intensity_diff):
            return float('Inf')

        return distance.cityblock(p1[0:2], p2[0:2])            

    def __get_cluster_labels(self, data):
        '''
            data: [[x, y, r]]
        '''
        return cluster.DBSCAN(self.__eps, self.__min_samples, metric=self.__dist, n_jobs=-1).fit_predict(data)

    def draw_bbxs(self, bbxs, image, color = (0, 255, 0)):
        '''
            bbxs: [[x1, y1, x2, y2]] | (x1, y1): top left corner, (x2, y2): bottom right corner
        '''
        clone = image.copy()
        for (x1, y1, x2, y2) in bbxs:
            cv2.rectangle(clone, (x1, y1), (x2, y2), color)

        return clone

    def crop_to_bbxs(self, bbxs, image):
        cropped_imgs = []
        for bbx in bbxs:
            cropped_imgs.append(image[bbx[1] : bbx[3] + 1, bbx[0] : bbx[2] + 1])

        return cropped_imgs
    
    def __make_dataset(self, reflectance_image):
        data = []
        
        for y in range(reflectance_image.shape[0]):
            for x in range(reflectance_image.shape[1]):
                if(reflectance_image[y, x] > 0):
                    data.append([x, y, reflectance_image[y, x]])
                
        return np.array(data)

    def get_bbxs(self, reflectance_image):
        data = self.__make_dataset(reflectance_image)
        clusters = self.__get_cluster_labels(data)
        bbxs = self.__get_bbx_from_clusters(data, clusters)

        return bbxs

    def __expand_bbxs(self, bbxs, expand_by = constants.EXPAND_BY):
        new_bbxs = []
        
        for bbx in bbxs:
            cushion_x, cushion_y = int(expand_by * (bbx[2] - bbx[0]) / 2), int(expand_by * (bbx[3] - bbx[1]) / 2)
            new_bbxs.append(np.array([bbx[0] - cushion_x, bbx[1] - cushion_y, bbx[2] + cushion_x, bbx[3] + cushion_y]))

        return new_bbxs
            
    def __get_bbx_from_clusters(self, data, clusters, expand_by = constants.EXPAND_BY, max_bbx_dist = constants.MAX_BBX_DIST):
        '''
            expand_by: percentage by which to inflate bounding box
        '''
        unique_clusters = np.unique(clusters)
        if(len(unique_clusters) == 1 and unique_clusters[0] == -1):
            raise self.NoClustersFound('no cluster found')

        bbxs = []
        for cluster_id in unique_clusters:
            if(cluster_id == -1):
                continue

            clustered_pts = data[clusters == cluster_id]
            
            #get original bounding box
            x1 = np.min(clustered_pts[:, 0])
            y1 = np.min(clustered_pts[:, 1])

            x2 = np.max(clustered_pts[:, 0])
            y2 = np.max(clustered_pts[:, 1])

            bbxs.append([x1, y1, x2, y2])

        bbxs = np.array(bbxs)
        bbxs = self.__merge_bbxs(bbxs, max_bbx_dist)
        bbxs = self.__filter_bbx_by_area(bbxs)

        #expand bounding box to make room for error and ease in detection
        return self.__expand_bbxs(bbxs, expand_by)
        
    def __merge_bbxs(self, bbxs, max_bbx_dist = constants.MAX_BBX_DIST):
        
        uf = union_find.UF(len(bbxs))
        for i in range(len(bbxs)):
            for j in range(i + 1, len(bbxs)):
                if(self.__dist_bbx(bbxs[i], bbxs[j]) < max_bbx_dist):
                    uf.union(i, j)

        labels = np.zeros(len(bbxs))
        
        for i in range(len(bbxs)):
            labels[i] = uf.find(i)
        
        unique_labels = np.unique(labels)
        new_bbxs = []
        for label in unique_labels:
            clustered_bbxs = bbxs[labels == label]
            new_bbxs.append([min(clustered_bbxs[:, 0]), min(clustered_bbxs[:, 1]), max(clustered_bbxs[:, 2]), max(clustered_bbxs[:, 3])])

        return new_bbxs

    def __dist_bbx(self, bbx1, bbx2):
        is_2_on_left    = bbx2[2] < bbx1[0]
        is_2_on_right   = bbx2[0] > bbx1[2] 
        is_2_on_bottom  = bbx2[1] > bbx1[3]
        is_2_on_top     = bbx2[3] < bbx1[1]

        if is_2_on_bottom:
            if is_2_on_right:
                return np.linalg.norm(bbx1[2:4] - bbx2[0:2])
            elif is_2_on_left:
                return np.linalg.norm(np.array([bbx1[0], bbx1[3]]) - np.array([bbx2[2], bbx2[1]]))
            else:
                return bbx2[1] - bbx1[3]
        elif is_2_on_top:
            if is_2_on_right:
                return np.linalg.norm(np.array([bbx1[2], bbx1[1]]) - np.array([bbx2[0], bbx2[3]]))
            elif is_2_on_left:
                return np.linalg.norm(bbx1[0:2] - bbx2[2:4])
            else:
                return bbx1[1] - bbx2[3]
        else:
            if is_2_on_right:
                return bbx2[2] - bbx1[0]
            elif is_2_on_left:
                return bbx1[0] - bbx2[2]
            else: #rectangles intersect
                return 0

    def __filter_bbx_by_area(self, bbxs, min_area = constants.MIN_AREA):
        new_bbxs = []
        for i, (x1, y1, x2, y2) in enumerate(bbxs):
            w = x2 - x1
            h = y2 - y1
            if(w * h > min_area):
                new_bbxs.append(bbxs[i])

        return new_bbxs

    def __display_clusters(self, data, clusters):
        import matplotlib.pyplot as plt
        import numpy as np

        unique_clusters = unique(clusters)
        colors = [plt.cm.Spectral(num) for num in np.linspace(0, 1, len(unique_clusters))]

        for cluster_id, color in zip(unique_clusters, colors):
            if(cluster_id == -1):
                color = [0, 0, 0, 1] #black for noise

            clustered_pts = data[clusters == cluster_id]

            for (x, y, r) in clustered_pts:
                rad = r * 20 / 255
                plt.plot(x, y, 'o', markerfacecolor=color, markeredgecolor='k', markersize=rad)

            plt.show()
