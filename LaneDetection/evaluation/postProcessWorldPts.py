#!/usr/bin/python3
import sys
import numpy as np
import math

fin = open(sys.argv[1])
fout = open(sys.argv[2], 'w')

ctrl_pts = 8
pts = np.array([np.array(line.split()).astype(np.float) for line in fin])
new_pts = pts[:ctrl_pts]

for i in range(ctrl_pts, len(pts), ctrl_pts):
    index_to_ins = np.searchsorted(new_pts[:, 0], pts[i, 0])
    new_pts = np.delete(new_pts, np.s_[index_to_ins:], axis=0)
    new_pts = np.append(new_pts, pts[i:i + ctrl_pts], axis=0)
else:
    avg_retention = math.ceil(len(new_pts) * ctrl_pts / len(pts))
    new_pts = np.delete(new_pts, np.s_[avg_retention-ctrl_pts:], axis=0)

for i, pt in enumerate(new_pts):
    for x in pt:
        fout.write(str(x) + '\t')
    if(i != len(new_pts) - 1):
        fout.write('\n')
