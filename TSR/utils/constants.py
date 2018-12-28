class _const:
    KITTI_BASE_DIR      = '/home/ankit/Research/Autonomous-Driving-Research/Datasets/KITTI'
    DATE                = '2011_09_29'
    DRIVE               = '0004'
    
    MIN_X               = 5
    CAM_NUM             = 2
    
    class ConstError(TypeError): pass
    def __setattr__(self, name, value):
        raise ConstError('attempt to change const value')

import sys
sys.modules[__name__] = _const