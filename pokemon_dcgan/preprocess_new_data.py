import os
import glob
import numpy as np

from matplotlib import pyplot as plt

from PIL import Image
from skimage.measure import label

import pdb


def pad_arr(arr):
    bg_val = sum(arr[0, 0])
    diff = max(arr.shape[:-1]) - min(arr.shape[:-1])
    diff_x, diff_y = int(np.floor(diff/2)), int(np.ceil(diff/2))
    idx_to_pad = np.argmin(arr.shape[:-1])

    if idx_to_pad == 0:
        pad_width = ((diff_x, diff_y), (0, 0), (0, 0))
    else:
        pad_width = ((0, 0), (diff_x, diff_y), (0, 0))

    # pdb.set_trace()
    pad = np.pad(arr, pad_width, 'constant', constant_values=bg_val)
    return pad


jpg_files = sorted(glob.glob("./data/new_data/*.JPG*"))
# png_files = sorted(glob.glob("./data/pokemon_/*.png*"))

for jpg in jpg_files:
# for jpg in png_files:
    jpg_im = Image.open(jpg)
    # pdb.set_trace()
    jpg_im = jpg_im.convert("RGBA")

    jpg_arr = np.array(jpg_im)
    if jpg_arr.shape[0] != jpg_arr.shape[1]:
        jpg_arr = pad_arr(jpg_arr)
    jpg_arr_ = np.sum(jpg_arr, axis=2)

    conn_region = label(jpg_arr_)

    bg = np.where(conn_region==1, 1, 0)
    bg = bg[:, :, np.newaxis]
    bg_re = np.repeat(bg, 4, axis=2)

    jpg_arr_nobg = np.where(bg_re, 0, jpg_arr)
    jpg_im_nobg = Image.fromarray(jpg_arr_nobg)

    if jpg_im_nobg.size[0] > 256:
        newsize = (256, 256)
        jpg_im_nobg_resize = jpg_im_nobg.resize(newsize)
        jpg_im_nobg_resize.save(os.path.join(os.path.dirname(jpg), os.path.basename(jpg)[:-4] + ".png"))
        # jpg_im.save(os.path.join(os.path.dirname(jpg), os.path.basename(jpg)[:-4] + ".png"))
        print(os.path.join(os.path.dirname(jpg), os.path.basename(jpg)[:-4] + ".png"), jpg_im_nobg_resize.size)
    # pdb.set_trace()

