import torch
import os
from PIL import Image
import ntpath

class TSRDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, single_class_transforms, multi_class_transforms, num_traffic_signs):
        super(TSRDataset, self).__init__()
        self.__img_files = []
        self.__num_traffic_signs = num_traffic_signs
        for img_file in sorted(os.listdir(root_dir)):
            self.__img_files.append(os.path.join(root_dir, img_file))
        self.__single_transforms = single_class_transforms
        self.__multi_transforms = multi_class_transforms

    def __len__(self):
        return len(self.__img_files)

    def __getitem__(self, idx):
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