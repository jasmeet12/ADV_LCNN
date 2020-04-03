import numpy as np
import scipy.io as io
import os
import glob


path = './logs/200305-125345-49f1b45-baseline/npz/000001200/'
out = []

for filename in os.listdir(path):
    if '.npz' not in filename:
        continue
    fpath = path+filename
    data = np.load(fpath)
    f_name = filename.split('.')[0]
    contours = []
    for i in range(data['lines'].shape[0]):
        lines = data['lines'][i]
        contours.append([lines[0][0],lines[0][1],lines[1][0],lines[1][1],data['score'][i]])

    mdict = {'filename':f_name,'contours':contours}
    out.append(mdict)
io.savemat('enetoend.mat', mdict={'out': out})



# GT = "data/wireframe/valid/*.npz"
# # gt_list = sorted(glob.glob(GT))
# #
# # out = []
# # for filename in gt_list:
# #     if '.npz' not in filename:
# #         continue
# #     fpath = filename
# #     data = np.load(fpath)
# #     f_name = filename.split('.')[0]
# #     contours = []
# #     for i in range(data['lines'].shape[0]):
# #         lines = data['lines'][i]
# #         contours.append([lines[0][0],lines[0][1],lines[1][0],lines[1][1],data['score'][i]])
# #
# #     mdict = {'filename':f_name,'contours':contours}
# #     out.append(mdict)
# #
# #
# #
# # io.savemat('gt_wireframe.mat', mdict={'out':out})
# io.loadmat('a1.mat')
