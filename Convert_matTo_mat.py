import numpy as np
import scipy.io as io
import os
import glob

GT = "/home/jasmeet/Documents/AdvanceComputerVision/lcnn/lcnn/data/wireframe/valid/*.mat"
gt_list = sorted(glob.glob(GT))

out = []
for i, filename in enumerate(gt_list):

    fpath = filename
    mat = io.loadmat(fpath)
    fname = filename.split('/')[-1].split('.')[0]
    mdict = {'filename':fname ,'contours':mat['lines']}

    out.append(mdict)

io.savemat('gt_wireframe.mat', mdict={'out':out})
# io.loadmat('a1.mat')
