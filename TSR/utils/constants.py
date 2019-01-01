import os

class _const:
    #general
    BASE_DIR            = '/home/ankit/Research/Autonomous-Driving-Research/'
    KITTI_BASE_DIR      = os.path.join(BASE_DIR, 'Datasets/KITTI')
    DATE                = '2011_09_29'
    DRIVE               = '0004'
    
    #velo_projector
    MIN_X               = 5
    CAM_NUM             = 2

    #bbx_generator
    EPS                 = 4
    MIN_PTS             = 7
    MIN_INTENSITY_DIFF  = 5
    EXPAND_BY           = 0.2
    MIN_AREA            = 25
    REFLECTIVITY_THRESH = int(0.6 * 255)
    MAX_BBX_DIST        = 20
        
    class ConstError(TypeError): pass
    def __setattr__(self, name, value):
        raise ConstError('attempt to change const value')

import sys
sys.modules[__name__] = _const