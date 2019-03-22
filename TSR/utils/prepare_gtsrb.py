#!/usr/bin/python3

import os
import random
import shutil

classes_file    = '/home/netbot/Research/Autonomous-Driving-Research/TSR/nn/classes.txt'
gtsrb_root      = '/home/netbot/Research/Autonomous-Driving-Research/Datasets/GTSRB/Final_Training/Images/'
train_dir       = '/home/netbot/Research/Autonomous-Driving-Research/Datasets/GTSRB/train/'
val_dir         = '/home/netbot/Research/Autonomous-Driving-Research/Datasets/GTSRB/val/'
test_dir        = '/home/netbot/Research/Autonomous-Driving-Research/Datasets/GTSRB/test/'

useful_classes = [0, 1, 2, 3, 4, 5, 13, 25, 14, 22, 43]
total_train_imgs = 19000
total_val_imgs   = 1000
total_test_imgs  = 7000
total_imgs       = total_test_imgs + total_val_imgs + total_train_imgs

train_count = 0
val_count = 0
test_count = 0

for idx, c in enumerate(useful_classes):
    cur_dir   = os.path.join(gtsrb_root, str(c).zfill(5))
    img_files = os.listdir(cur_dir)
    random.shuffle(img_files)
    img_files = [ f for f in img_files if os.path.splitext(f)[1] in ['.png', '.jpg', '.ppm']]
    
    num_train   = int(total_train_imgs / len(useful_classes))
    num_val     = int(total_val_imgs / len(useful_classes))
    num_test    = int(total_test_imgs / len(useful_classes))

    train = []
    val   = []
    test  = []

    if(num_train + num_val + num_test > len(img_files)):
        num_train   = int(total_train_imgs / total_imgs * len(img_files))
        num_val     = int(total_val_imgs / total_imgs * len(img_files))
        num_test    = int(total_test_imgs / total_imgs * len(img_files))
    
    train = img_files[:num_train]
    val   = img_files[num_train:num_train + num_val]
    test  = img_files[num_train + num_val:]

    for img_file in train:
        shutil.copyfile(os.path.join(cur_dir, img_file), os.path.join(train_dir, str(idx).zfill(2) + '_' + str(train_count).zfill(6) + os.path.splitext(img_files[0])[1]))
        train_count += 1

    for img_file in val:
        shutil.copyfile(os.path.join(cur_dir, img_file), os.path.join(val_dir, str(idx).zfill(2) + '_' + str(val_count).zfill(6) + os.path.splitext(img_files[0])[1]))
        val_count += 1

    for img_file in test:
        shutil.copyfile(os.path.join(cur_dir, img_file), os.path.join(test_dir, str(idx).zfill(2) + '_' + str(test_count).zfill(6) + os.path.splitext(img_files[0])[1]))
        test_count += 1
