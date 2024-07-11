import os
import sys
import glob
import cv2
import pathlib
import numpy as np
from tqdm.auto import tqdm

KANJI_NPZ='ETL8G_64x64.npz'

def to64x64(img):
    h, w = img.shape[0], img.shape[1]
    if h > w:
        img2 = np.zeros((h,h), dtype=np.uint8)
        img2[:, (h-w)//2:(h-w)//2+w] = img
    elif w<h:
        img2 = np.zeros((w,w), dtype=np.uint8)
        img2[(w-h)//2:(w-h)//2+h,:] = img
    else:
        img2 = img
    return cv2.resize(img2, (64, 64))

def get_bg(prefix):
    bg = []
    for j in tqdm(list(glob.glob(os.path.join(prefix, '**', '*.jpg')))):
        b = os.path.splitext(j)[0] + '.bbox'
        assert os.path.exists(b)

        # n, x, y, w, h
        bbox = np.loadtxt(b, dtype=np.int32)
        assert bbox.shape == (5,)
        assert bbox[0] == 1

        # kanji region
        img = cv2.imread(j, cv2.IMREAD_GRAYSCALE)
        bgval = np.median(img)
        _, x, y, w, h = bbox

        # bgsub + inv
        kanji = img[y:y+h,x:x+w]
        kanji[ kanji > bgval - 5 ] = 255
        kanji = 255 - kanji

        # resize to 64x64
        kanji = to64x64(kanji)

        bg.append(kanji)
    return np.array(bg)


def mk300x300(fg, bgs):
    assert fg.shape == (64,64)
    assert bgs.shape[1] == 64
    assert bgs.shape[2] == 64

    # select 9 kanjis: 8 bg + 1 fg
    k = bgs[np.random.randint(0, bgs.shape[0], 9), :, :]
    k[0, :, :] = fg

    # shuffle
    idx = np.random.permutation(k.shape[0])
    k = k[idx,:,:]

    # embed with random jitter
    img = np.zeros((300,300), dtype=np.uint8)
    bbox = np.zeros(4, dtype=np.int32)
    bbox_k = None
    for i in range(9):
        r = i//3
        c = i%3

        # random top-left corner
        dx, dy = np.random.randint(0, 100-64, 2)
        bx = c*100+dx
        by = r*100+dy

        # random resize
        mag = np.random.uniform(0.5, 1.5)
        sz = int(64 * mag)
        img2 = cv2.resize(k[i], (sz, sz))

        if bx + sz > img.shape[1]:
            bx = img.shape[1] - sz
        if by + sz > img.shape[0]:
            by = img.shape[0] - sz

        # embed
        img[by:by+sz,bx:bx+sz] = img2

        # save bbox
        if idx[i] == 0:
            bbox[0] = bx
            bbox[1] = by
            bbox[2] = sz
            bbox[3] = sz
            bbox_k = np.copy(img2)

    # make fg to be the front-most image
    img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = bbox_k

    return img, bbox


print(f'loading {KANJI_NPZ}')

kanji64 = np.load(KANJI_NPZ)['img']
moji = np.load(KANJI_NPZ)['moji']
hira = np.load(KANJI_NPZ)['hira']
label = np.load(KANJI_NPZ)['label']

h_idx = (hira == 1)
h_img = kanji64[h_idx,:,:]
h_moji = moji[h_idx]
h_label = label[h_idx]

k_idx = (hira == 0)
k_img = kanji64[k_idx,:,:]
k_moji = moji[k_idx]
k_label = label[k_idx]

out_img = []
out_bbox = []
out_moji = []
out_label = []

for i in tqdm(range(h_img.shape[0])):
    hi = h_img[i,:,:]
    hm = h_moji[i]
    hl = h_label[i]

    img300, bbox300 = mk300x300(hi, k_img)
    out_img.append(img300)
    out_bbox.append(bbox300)
    out_moji.append(hm)
    out_label.append(hl)

out_img = np.array(out_img)
out_bbox = np.array(out_bbox)
out_moji = np.array(out_moji)
out_label = np.array(out_label)

np.savez_compressed('ETL8G_300x300.npz', img=out_img, bbox=out_bbox, moji=out_moji, label=out_label)

"""
for dirname in ['train', 'val']:
    imgdir = os.path.join(PREFIX, f'{dirname}2017')
    os.makedirs(imgdir, exist_ok=True)

    with open(os.path.join(PREFIX, 'annotations', f'instances_{dirname}2017.json'), 'w') as fp:
        out_file_name = []
        out_height = []
        out_width = []
        out_bbox = []
        out_category = []
        out_area = []
        
        for f in tqdm(sorted(glob.glob(f'ETL8G_KANA/{dirname}/*/????_????.png'))):
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            k, ret, bbox = kanji.crop(img, 1, dist_th=100)
            if ret != 0:
                print(f'[BUG] cannot find a character in {f}')
                continue
        
            # bgsub + inv
            #bgval = np.median(img)
            #k[ k > bgval - 5 ] = 255
            #k = 255 - k

            # random position
            img300, bbox300 = mk300x300(k, kanji64)

            out_area.append( bbox[2] * bbox[3] )
            out_file_name.append(os.path.basename(f))
            out_bbox.append(bbox300)
            out_category.append(ord(pathlib.Path(f).parts[2]) - ord('あ') + 1)
            
            cv2.imwrite(f'{imgdir}/{os.path.basename(f)}', img300)

            out_height.append(img300.shape[0])
            out_width.append(img300.shape[1])

            #print(ret, bbox)
        
            #buf = cv2.rectangle(img300, (bbox300[0], bbox300[1]), (bbox300[0]+bbox300[2], bbox300[1]+bbox300[3]), (0, 255, 0), 3)
            #cv2.imshow("hoge", buf)
            #cv2.waitKey(0)
        
        
        fp.write(f'{{"images": [\n')
        eol = ','
        for id, (file_name, height, width) in enumerate(zip(out_file_name, out_height, out_width), start=1):
            if id == len(out_file_name):
                eol = ''
            fp.write(f'{{"file_name": "{file_name}","height": {height},"width": {width},"id": {id}}}{eol}\n')
        fp.write(f'],\n')
        
        fp.write(f'"annotations": [\n')
        eol = ','
        for id, (bbox, category, area) in enumerate(zip(out_bbox, out_category, out_area), start=1):
            if id == len(out_bbox):
                eol = ''
            fp.write(f'{{"image_id": {id},"bbox": [{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}],"category_id": {category},"id": {id},"iscrowd": 0,"area": {area}}}{eol}\n')
        fp.write(f'],\n')
        
        categories = sorted(set(out_category), key=out_category.index)
        fp.write(f'"categories": [\n')
        eol = ','
        for id, category in enumerate(categories, start=1):
            if id == len(categories):
                eol = ''
            name = chr(category + ord('あ') - 1)
            fp.write(f'{{"supercategory": "hiragana","id": {category},"name": "{name}"}}{eol}\n')
        fp.write(f']}}\n')

"""