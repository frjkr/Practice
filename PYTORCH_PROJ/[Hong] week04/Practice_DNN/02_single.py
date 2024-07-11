import os
import sys
import numpy as np
import cv2
import glob
import pathlib
from tqdm.auto import tqdm

SZ = 64
N = 160

# get subdirs
out_img = []
out_moji = []
out_hira = []

# https://ja.wikipedia.org/wiki/%E5%B9%B3%E4%BB%AE%E5%90%8D_(Unicode%E3%81%AE%E3%83%96%E3%83%AD%E3%83%83%E3%82%AF)
def is_hiragana(moji):
    return 0x3041 <= ord(moji) <= 0x309F

pbar = tqdm(sorted(glob.glob('ETL8G_PNG/SRC/*/')), desc='？')
for i in pbar:
    p = pathlib.Path(i)
    moji = p.name
    hira = is_hiragana(moji)
    pbar.set_description(moji)
    #print(moji, hira)

    for count, j in enumerate(glob.glob(f'{i}/*.png')):
        if count == N:
            break
        img = cv2.imread(j, cv2.IMREAD_GRAYSCALE)
        img = 255 - cv2.resize(img, (SZ, SZ), interpolation=cv2.INTER_AREA)
        out_img.append(img)
        out_moji.append(moji)
        out_hira.append(hira)

out_img = np.array(out_img)
out_moji = np.array(out_moji)
out_hira = np.array(out_hira)

# 文字集合
m = sorted(list(set(out_moji)))
# 連番に変換
moji2idx = dict(zip(m, range(len(m))))
out_label = np.array([moji2idx[i] for i in out_moji])

print(out_img.shape)
print(out_moji.shape)
print(out_hira.shape)
print(out_label.shape)

np.savez_compressed(f'ETL8G_{SZ}x{SZ}.npz', img=out_img, moji=out_moji, hira=out_hira, label=out_label)
