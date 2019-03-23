import torch
import os
from PIL import Image
import ntpath
from utils import constants

class TSRDataset(torch.utils.data.Dataset):

    def __init__(self, single_class_transforms, multi_class_transforms, num_traffic_signs = constants.NUM_TRAFFIC_SIGNS, root_dir = None, img_list = None):
        super(TSRDataset, self).__init__()
        self.__num_traffic_signs = num_traffic_signs
        self.__img_list = None
        if(img_list):
            self.__img_list = img_list
        else:
            self.__img_files = []
            for img_file in sorted(os.listdir(root_dir)):
                self.__img_files.append(os.path.join(root_dir, img_file))
        
        self.__single_transforms = single_class_transforms
        self.__multi_transforms = multi_class_transforms

    def __len__(self):
        if(self.__img_list):
            return len(self.__img_list)
        else:
            return len(self.__img_files)

    def __getitem__(self, idx):
        if(self.__img_list):
            img = self.__img_list[idx]
            img = self.__single_transforms(img)
            if(torch.cuda.is_available()):
                img = img.to(torch.device('cuda'))

            return img
        else:
            img = Image.open(self.__img_files[idx]).convert('RGB')
            labels = list(map(int, ntpath.basename(self.__img_files[idx]).split('_')[:-1]))
            gt_class_probs = torch.zeros(self.__num_traffic_signs)
            gt_class_probs[labels] = 1
            
            if len(labels) == 1 and self.__single_transforms:
                img = self.__single_transforms(img)
            elif len(labels) > 1 and self.__multi_transforms:
                img = self.__multi_transforms(img)

            if(torch.cuda.is_available()):
                img = img.to(torch.device('cuda'))
                gt_class_probs = gt_class_probs.to(torch.device('cuda'))

            return img, gt_class_probs