#!/usr/bin/python3

import os
import random
import shutil
from pathlib import Path

base_dir		= str(Path(os.path.realpath(__file__)).parents[2])
gtsrb_root      = os.path.join(base_dir, 'Datasets/GTSRB/Final_Training/Images/')
train_dir       = os.path.join(base_dir, 'Datasets/GTSRB/train/')
val_dir         = os.path.join(base_dir, 'Datasets/GTSRB/val/')
test_dir        = os.path.join(base_dir, 'Datasets/GTSRB/test/')

useful_classes   = [0, 1, 2, 3, 4, 5, 13, 25, 14, 22, 43]
total_train_imgs = 9
total_val_imgs   = 1
total_test_imgs  = 0
total_imgs       = total_test_imgs + total_val_imgs + total_train_imgs

train_count = 0
val_count = 0
test_count = 0

for idx, c in enumerate(useful_classes):
    cur_dir   = os.path.join(gtsrb_root, str(c).zfill(5))
    img_files = os.listdir(cur_dir)
    random.shuffle(img_files)
    img_files = [ f for f in img_files if os.path.splitext(f)[1] in ['.png', '.jpg', '.ppm']]
    print(len(img_files))
#    num_train   = int(total_train_imgs / len(useful_classes))
#    num_val     = int(total_val_imgs / len(useful_classes))
#    num_test    = int(total_test_imgs / len(useful_classes))

    train = []
    val   = []
    test  = []

    num_train   = int(total_train_imgs / total_imgs * len(img_files))
    num_val     = int(total_val_imgs / total_imgs * len(img_files))
    num_test    = int(total_test_imgs / total_imgs * len(img_files))
    
    train = img_files[:num_train]
    
    if(num_val > 0):
        val = img_files[num_train:num_train + num_val]
    else:
        val = []

    if(num_test > 0):
        test = img_files[num_train + num_val:]
    else:
        test = []

    for img_file in train:
        shutil.copyfile(os.path.join(cur_dir, img_file), os.path.join(train_dir, str(idx).zfill(2) + '_' + str(train_count).zfill(6) + os.path.splitext(img_files[0])[1]))
        train_count += 1

    for img_file in val:
        shutil.copyfile(os.path.join(cur_dir, img_file), os.path.join(val_dir, str(idx).zfill(2) + '_' + str(val_count).zfill(6) + os.path.splitext(img_files[0])[1]))
        val_count += 1

    for img_file in test:
        shutil.copyfile(os.path.join(cur_dir, img_file), os.path.join(test_dir, str(idx).zfill(2) + '_' + str(test_count).zfill(6) + os.path.splitext(img_files[0])[1]))
        test_count += 1
