import os
import numpy as np
from skimage import io, exposure
from skimage.transform import resize
# import matplotlib.pyplot as plt

def make_lungs():
    path = 'scr/scratch/fold1/landmarks/'
    for i, filename in enumerate(os.listdir(path)):
        img = 1.0 - np.fromfile(path + filename, dtype='>u2')
        img2=resize(img,(256,256))
        io.imsave('new/image'+str(i)+'.png',img2)
        print(img.shape)

        # img = 1.0 - np.fromfile(path + filename, dtype='>u2').reshape((2048, 2048)) * 1. / 4096
        # img = exposure.equalize_hist(img)
        # io.imsave('new/image/' + filename[:-4] + '.png', img)
        # print ('Lung', i, filename)

def make_masks():
    path = 'scr/scratch/fold1/masks/left lung/'
    for i, filename in enumerate(os.listdir(path)):
        left = io.imread('scr/scratch/fold1/masks/left lung/' + filename[:-4] + '.gif')
        right =io.imread('scr/scratch/fold1/masks/right lung/' + filename[:-4] + '.gif')
        io.imsave('new/mask/' + filename[:-4] + 'msk.png', np.clip(left + right, 0, 255))
        print ('Mask', i, filename)

make_lungs()
# make_masks()
