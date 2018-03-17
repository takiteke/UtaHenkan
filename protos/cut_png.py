import numpy as np
from tqdm import tqdm

from PIL import Image


dataDir='../input/'
tar_width = 137
first_index = 0
index = first_index
for i in tqdm(range(0, 40)):
    miki = Image.open(dataDir+"png/%02d_miki.png"%i)
    ritsuko = Image.open(dataDir+"png/%02d_ritsuko.png"%i)

    w = miki.size[0]
    cut_num = w // tar_width
    for j in range(cut_num):
        #cut
        miki_cut = miki.crop((j*tar_width, 0, min((j+1)*tar_width, w), 513))
        ritsuko_cut = ritsuko.crop((j*tar_width, 0, min((j+1)*tar_width, w), 513))

        #cutしたデータをpngで保存
        miki_cut.save(dataDir+"png_cut/%04d_miki.png"%index)
        ritsuko_cut.save(dataDir+"png_cut/%04d_ritsuko.png"%index)

        index += 1