import os
from pathlib import Path
from torchvision import transforms
import torch

class ToDefaultTensor():
    def __call__(self, x):
        return transforms.ToTensor()(x).cuda() if torch.cuda.is_available() else transforms.ToTensor()(x)

class _const:
    #general
    BASE_DIR            = str(Path(os.path.realpath(__file__)).parents[2])
    KITTI_BASE_DIR      = os.path.join(BASE_DIR, 'Datasets/KITTI')
    RESULTS_BASE_DIR    = os.path.join(BASE_DIR, 'Results')
    DATE                = '2011_09_29'
    DRIVE               = '0004'
    
    #velo_projector
    MIN_X               = 5
    CAM_NUM             = 2

    #bbx_generator
    EPS                 = 3
    MIN_PTS             = 15
    MIN_INTENSITY_DIFF  = 5
    EXPAND_BY           = 0.5
    MIN_AREA            = 200
    REFLECTIVITY_THRESH = int(0.6 * 255)
    MAX_BBX_DIST        = 20

    #map_gen_solver
    SLAM_OP_FILE_NAME   = 'slam_op.txt'
    MAP_FILE_PREFIX     = 'map_'

    #nn
    PRINT_FREQ          = 20
    TENSORBOARD_LOG_DIR = os.path.join(BASE_DIR, 'TSR/nn/runs/')
    LR                  = 0.01
    MOMENTUM            = 0.9
    NUM_TRAFFIC_SIGNS   = 11
    BATCH_SIZE          = 256
    GTSRB_ROOT          = os.path.join(BASE_DIR, 'Datasets/GTSRB')
    THRESHOLD_PROB      = 0.85
    TRAIN_EPOCHS        = 80
    MEAN                = [0.37043, 0.32912, 0.32900]
    STD                 = [0.17315, 0.17867, 0.19009]
    IN_SIZE             = [52, 52]
    BEST_MODEL_PATH     = os.path.join(BASE_DIR, 'TSR/nn/best.pth')
    
    TRAIN_TRANSFORMS_SINGLE_CLASS = transforms.Compose(
                                        [
                                            transforms.RandomChoice([
                                                transforms.CenterCrop(IN_SIZE),
                                                transforms.Resize(IN_SIZE),
                                                transforms.RandomCrop(IN_SIZE, pad_if_needed=True)
                                            ]),
                                            transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)], 0.2),
                                            ToDefaultTensor(),
                                            transforms.Normalize(MEAN, STD)
                                        ]
                                    )
    TRAIN_TRANSFORMS_MULTI_CLASS  = transforms.Compose(
                                        [        
                                            transforms.Resize(IN_SIZE),
                                            transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)], 0.2),
                                            ToDefaultTensor(),
                                            transforms.Normalize(MEAN, STD)
                                        ]
                                    )
    TEST_TRANSFORMS               = transforms.Compose(
                                        [                         
                                            transforms.Resize(IN_SIZE),
                                            ToDefaultTensor(),
                                            transforms.Normalize(MEAN, STD)
                                        ]
                                    )

    class ConstError(TypeError): pass
    def __setattr__( name, value):
        raise ConstError('attempt to change const value')

import sys
sys.modules[__name__] = _const
