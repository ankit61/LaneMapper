import velo_projector

class PtsExtractor(velo_projector.VeloProjector):
    def __init__(self, calib, cam_num = constants.CAM_NUM, min_x = constants.MIN_X):
        velo_projector.VeloProjector.__init__(calib, cam_num, min_x)
    
    def extract_pts_inside_bbx(self, bbx, img_pts, lidar_pts):
        '''
            bbx = (x, y, w, h), (x, y) = top left corner
            img_pts = (x, y, r) or (x, y)
            lidar_pts = (x, y, z, r) or (x, y, z)

            img_pts and lidar_pts indices match: i.e. at index i, img_pts[i] represents the projection of lidar_pts[i]
        '''
        #FIXME: can improve speed by using fancy data structures
        pts_in_bbx = []
        for i, pt in enumerate(img_pts):
            if(self.__inside_box(bbx, pt)):
                pts_in_bbx.append(lidar_pts[i])
        
        return pts_in_bbx

    def __inside_box(self, bbx, pt):
        x = pt[0]
        y = pt[1]

        return ((x >= bbx[0]) and (x <= (bbx[0] + bbx[2])) and (y >= bbx[1]) and (y <= bbx[1] + bbx[3]))