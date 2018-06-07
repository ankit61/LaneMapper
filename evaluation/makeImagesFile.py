from os import listdir
from os.path import isfile, join
import sys
imageFile = open(sys.argv[2], 'w')
for f in listdir(sys.argv[1]):
    if isfile(join(sys.argv[1], f)):
        imageFile.write(f + "\n")


