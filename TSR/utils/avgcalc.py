#!/usr/bin/python3
import cv2
import torch
import os
import numpy as np
import sys

path = '/home/netbot/Research/Autonomous-Driving-Research/Datasets/GTSRB/train/'#sys.argv[1]

image_names = os.listdir(path) 

mean = np.zeros((3,1))
std = np.zeros((3,1))
size = np.zeros(2)

length = len(image_names)

print("Output is displayed in BGR fashion, not RGB like pytorch expects")

for n in range(0, length):
	im = cv2.imread(os.path.join(path + image_names[n]))
	if(im is None):
		raise Exception("incorrect path: " + os.path.join(path + image_names[n]))
	temp  = cv2.meanStdDev(im)	  
	
	mean += temp[0]
	std	 += temp[1]
	size += np.array(im.shape[:-1])

mean = 	mean / (length * 255.0) 

std  = 	std / (length * 255.0)

size = 	size / length

print("mean:", mean)
print("std:", std)
print("size:", size)
