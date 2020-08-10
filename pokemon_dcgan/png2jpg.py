import os
import glob
import numpy as np

from matplotlib import pyplot as plt

from PIL import Image
from skimage.measure import label

import pdb

png_files = sorted(glob.glob("./data/pokemon/*.png*"))

for png in png_files:
    png_im = Image.open(png)
    png_im.load()

    background = Image.new("RGB", png_im.size, (255, 255, 255))
    background.paste(png_im, mask=png_im.split()[3])  # 3 is the alpha channel

    background.save(os.path.join(os.path.dirname(png), os.path.basename(png)[:-4] + ".jpg"), 'JPEG', quality=100)
    print(os.path.join(os.path.dirname(png), os.path.basename(png)[:-4] + ".jpg"), background.size)
    # pdb.set_trace()