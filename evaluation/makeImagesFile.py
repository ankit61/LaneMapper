#!/usr/bin/env python
#This script provides a very handy way to put image names into a file.
#It takes two inputs the directory which stores all images whose names you want to be put in a file and the name of file where these image names should be put.
#It assumes all files in the first directory are images only. By trivial changes, you can force it to only put relevant images
#Instructions to run: python3 makeImagesFile.py name-of-directory-where-images-are name-of-file-to-put-image-names
#This is very useful to generate the file that'll be passed as input to Evaluate.cpp
from os import listdir
from os.path import isfile, join
import sys
imageFile = open(sys.argv[2], 'w')
for f in sorted(listdir(sys.argv[1])):
    if isfile(join(sys.argv[1], f)):
        imageFile.write(f + "\n")
