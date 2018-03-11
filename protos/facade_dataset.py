import os

import numpy
from PIL import Image

import numpy as np

from chainer.dataset import dataset_mixin

# download `BASE` dataset from http://cmp.felk.cvut.cz/~tylecr1/facade/
class FacadeDataset(dataset_mixin.DatasetMixin):
    def __init__(self, dataDir='../input/png_cut', data_range=(0,120)):
        print("load dataset start")
        print("    from: %s"%dataDir)
        print("    range: [%d, %d)"%(data_range[0], data_range[1]))
        self.dataDir = dataDir
        self.dataset = []
        for i in range(data_range[0],data_range[1]):
            img = Image.open(dataDir+"/%04d_miki.png"%i)
            label = Image.open(dataDir+"/%04d_ritsuko.png"%i)

            img = np.asarray(img).astype("f")/128.0-1.0
            label = np.asarray(label).astype("f")/128.0-1.0

            img = img.transpose(2, 0, 1)[:2,:,:]
            label = label.transpose(2, 0, 1)[:2,:,:]

            self.dataset.append((img,label))
        print("load dataset done")
    
    def __len__(self):
        return len(self.dataset)
    
    # return (label, img)
    def get_example(self, i, crop_width=256):
        crop_width_h = int(crop_width * 2)
        crop_width_w = int(crop_width / 2)
        _,h,w = self.dataset[i][0].shape
        x_l = np.random.randint(0,w-crop_width_w)
        x_r = x_l+crop_width_w
        y_l = np.random.randint(0,h-crop_width_h)
        y_r = y_l+crop_width_h
        return self.dataset[i][1][:,y_l:y_r,x_l:x_r], self.dataset[i][0][:,y_l:y_r,x_l:x_r]