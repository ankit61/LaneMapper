import os
import solver
from utils import constants
from nn import net, dataset
import bbx_generator, lidar_image_generator
import torch

class TSRDetectSolver(solver.Solver):
    def __init__(self, base_dir = constants.KITTI_BASE_DIR, date = constants.DATE, drive = constants.DRIVE):
        solver.Solver.__init__(self, base_dir, date, drive)
        self.__net              = net.TSRNet()
        self.__bbx_gen          = bbx_generator.BbxGenerator()
        self.__lidar_img_gen    = lidar_image_generator.LIDARImageGenerator(self._dataset.calib)
        self.__output_file      = open(os.path.join(constants.BASE_DIR, 'TSR/traffic_signs.txt'), 'w')
        classes_file            = open(os.path.join(constants.BASE_DIR, 'TSR/nn/classes.txt'))
        self.__classes          = [classes_file.readline().strip() for _ in range(constants.NUM_TRAFFIC_SIGNS)]

    def run_nn(self, img_list):
        nn_dataset = dataset.TSRDataset(single_class_transforms = constants.TEST_TRANSFORMS, multi_class_transforms = constants.TEST_TRANSFORMS, img_list = img_list)
        nn_loader  = torch.utils.data.DataLoader(
            nn_dataset, batch_size = constants.BATCH_SIZE
        )
        pred_ts = {}
        with torch.no_grad():
            for imgs in nn_loader:
                probs = self.__net(imgs)

                list(map(lambda x, pred_ts = pred_ts, probs = probs : pred_ts.update( { x[1].item() : probs[(x[0].item(), x[1].item())].item() } ), (probs >= constants.THRESHOLD_PROB).nonzero()))

        return pred_ts

    def solve(self, cv2_img, velo, base_filename):
        lidar_img = self.__lidar_img_gen.generate_refined(velo, cv2_img.shape)
        bbxs = self.__bbx_gen.get_bbxs(lidar_img)
        pil_img = self.cv2_to_pil(cv2_img)
        
        if(bbxs):
            potential_sign_imgs = []
            for bbx in bbxs:
                potential_sign_imgs.append(pil_img.crop(tuple(bbx)))
     
            pred_ts = self.run_nn(potential_sign_imgs)

            if(pred_ts):
                self.__output_file.write(base_filename + ': ')
                for ts in pred_ts:
                    self.__output_file.write('(' + str(self.__classes[ts]) + ', ' + str(pred_ts[ts]) + ')\t')
                self.__output_file.write('\n')
                self.__output_file.flush()