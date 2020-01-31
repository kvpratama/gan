import glob
import random
import os
import numpy as np
from math import ceil, floor

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import scipy.ndimage

import matplotlib.pyplot as plt
import pdb
import warnings


class ImageDataset(Dataset):
    def __init__(self, root, opt, imsize=256, mode="train", out_channels=3):
        self.imsize = imsize
        self.out_channels = out_channels
        # self.sketch = sketch
        self.opt = opt

        self.files_data = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        # self.resize_input = resize_input

        self.data_min_val = 0
        self.data_max_val = 255

        self.opt.augmentation = False if mode == "test" else self.opt.augmentation

    def __getitem__(self, index):
        img = Image.open(self.files_data[index]).convert('RGBA')
        # print(np.array(img).dtype)
        mask = np.where(np.array(img)[:, :, 3], 0, 1)
        img = np.array(img)[:, :, :3]

        # get mask start and end index
        mask_index = np.where(mask == 0)
        row_min = mask_index[0].min()
        row_max = mask_index[0].max()
        col_min = mask_index[1].min()
        col_max = mask_index[1].max()

        if self.opt.augmentation:
            self.opt.sigma = np.random.randint(5, 20)
            self.opt.threshold = np.random.randint(150, 255)

        if self.opt.edges_to_sketch:
            img = self.sketchify(img, self.files_data[index], self.opt.sigma, threshold=self.opt.threshold)
            img = np.where(mask, 255, img)  # remove background noise after sketchify image
            # img = Image.fromarray(img)
            # img = img.convert('1')
            # plt.imshow(img, cmap='gray')
            # plt.show()
            # img = np.array(img)

        elif self.opt.sketch_to_color:
            sketch = self.sketchify(img, self.files_data[index], self.opt.sigma, threshold=self.opt.threshold)
            img = np.where(np.expand_dims(mask, axis=2).repeat(3, axis=2), 255, img)  # remove background noise
            mask = np.where(mask, 255, sketch)  # remove background noise after sketchify image

        mask = Image.fromarray(mask)
        img = Image.fromarray(img)

        img_data_ratio = self.resize_keep_ratio(img, self.imsize)
        mask_ratio = self.resize_keep_ratio(mask, self.imsize)

        transform = transforms.Compose([])
        if self.opt.augmentation:
            rot_angle = np.random.randint(-30, 30)
            translate_h, translate_v = np.random.randint(0, 30, size=2)
            scale_ = np.random.uniform(low=0.85, high=1.25)
            shear_angle = np.random.randint(-15, 15)
            transform = transforms.Compose([
                # transforms.ColorJitter(hue=0.25),
                # transforms.ColorJitter(saturation=1, hue=0.5),
                # transforms.RandomAffine(30, scale=(1, 1.2), fillcolor=(0,0,0,1))])
                # transforms.RandomAffine(30)
            ])
            img_data_ratio = transforms.functional.affine(img_data_ratio, angle=rot_angle, translate=(translate_h, translate_v),
                                         scale=scale_, shear=shear_angle, fillcolor=(255, 255, 255))
            mask_ratio = transforms.functional.affine(mask_ratio, angle=rot_angle, translate=(translate_h, translate_v),
                                         scale=scale_, shear=shear_angle, fillcolor=255)

        img_data_ratio_cj = transform(img_data_ratio)
        mask_ratio_cj = transform(mask_ratio)

        img_data_pad = self.pad_image(img_data_ratio_cj, self.imsize)
        mask_pad = self.pad_image(mask_ratio_cj, self.imsize)

        img_data_ratio_pad = img_data_pad(img_data_ratio_cj)
        mask_ratio_pad = mask_pad(mask_ratio)

        # input_mask = img_data_ratio_pad[3:, :, :]
        input_mask = mask_ratio_pad

        if self.opt.reverse:
            return {"data": img_data_ratio_pad, "gt": input_mask, "img": np.array(img),
                    'path': self.files_data[index], 'mask_index': [row_min, row_max, col_min, col_max]}
        else:
            return {"data": input_mask, "gt": img_data_ratio_pad, "img": np.array(img),
                    'path': self.files_data[index], 'mask_index': [row_min, row_max, col_min, col_max]}

    def __len__(self):
        return len(self.files_data)

    def pad_image(self, img, target_size):
        old_size = img.size
        pad_size_w = (target_size - old_size[0]) / 2
        pad_size_h = (target_size - old_size[1]) / 2

        if pad_size_w % 2 == 0:
            wl, wr = int(pad_size_w), int(pad_size_w)
        else:
            wl = ceil(pad_size_w)
            wr = floor(pad_size_w)

        if pad_size_h % 2 == 0:
            ht, hb = int(pad_size_h), int(pad_size_h)
        else:
            ht = ceil(pad_size_h)
            hb = floor(pad_size_h)

        return transforms.Compose(
            [
                transforms.Pad((wl, ht, wr, hb), fill=0),
                transforms.ToTensor(),
            ]
        )

    def resize_keep_ratio(self, img, target_size):
        old_size = img.size  # old_size[0] is in (width, height) format

        ratio = float(target_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # im = img.resize(new_size, Image.LANCZOS)
        im = img.resize(new_size)

        return im

    def sketchify(self, img, path, sigma=1, threshold=250):
        grayscale = self.grayscale(img)
        invert = 255 - grayscale
        blur = scipy.ndimage.filters.gaussian_filter(invert, sigma=sigma)
        sketch = self.dodge(blur, grayscale, path)
        # sketch = np.where(sketch < threshold, 0, sketch)
        sketch = np.where(sketch < threshold, 0, 255)
        # plt.imshow(sketch, cmap='gray')
        # plt.show()
        return sketch

    def grayscale(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def dodge(self, front, back, path):
        # warnings.filterwarnings("error")
        # result = front
        # try:
        result = front * 255 / (255 - back)
        result[result > 255] = 255
        result[back == 255] = 255
        # except:
        #     print(path)
        return result.astype('uint8')
