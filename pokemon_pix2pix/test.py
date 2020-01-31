from models import GeneratorUNet
from datasets import ImageDataset
import torch
from torch.autograd import Variable
import argparse
import os
import glob
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from PIL import Image
import pydicom
import numpy as np
import math
import nibabel as nib
from skimage.measure import compare_ssim

import pdb
import matplotlib.pyplot as plt


# Example command
# python test.py --checkpoint_model_dir "D:\pytorch_gan\srgan_xray\xray\saved_models" --test_dir_path "E:\data\xray\20190828\final_dicom\test"

def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)


def psnr(mse, pixel_max=1.0):
    if mse == 0:
        return 100
    return 20 * math.log10(pixel_max / math.sqrt(mse))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir_path", type=str, default='.\\data', help="Path to test image dir")
    parser.add_argument("--name", type=str, default="pokemon", help="name of the project")
    parser.add_argument("--generator", type=str, default="1", help="name of the generator")
    parser.add_argument("--checkpoint_model_dir", type=str, default='.\\checkpoints', help="Path to checkpoint model dir")
    # parser.add_argument("--output_dir", type=str, default=".\\checkpoints", help="Output directory")
    opt = parser.parse_args()
    print(opt)

    # checkpoints = [c for c in glob.glob(opt.checkpoint_model_dir + "\\*generator_90*")]
    checkpoints = [c for c in glob.glob(os.path.join(opt.checkpoint_model_dir, opt.name, 'saved_models', '*generator_' + opt.generator + '*'))]
    # filenames_all = [f for f in glob.glob(opt.test_dir_path + "\\*.*")]

    # checkpoint_metric = []
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    for checkpoint in checkpoints:
        checkpoint_name = checkpoint.split("\\")[-1].split(".")[0]
        os.makedirs("%s/%s/%s/%s" % (opt.checkpoint_model_dir, opt.name, 'outputs', checkpoint_name), exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define model and load model checkpoint
        state = torch.load(checkpoint, map_location=device)
        train_opt = state['opt']
        in_channels = train_opt.in_channels
        out_channels = train_opt.out_channels
        imsize = train_opt.imsize
        min_val_data = state['min_val_data']
        max_val_data = state['max_val_data']
        generator = GeneratorUNet(in_channels=in_channels, out_channels=out_channels).to(device)
        generator.load_state_dict(state.get('weight', False))
        generator.eval()

        # pdb.set_trace()
        # Temp custom edit of opt
        # train_opt.sigma = 5
        # train_opt.sketch_to_color = False

        dataloader = DataLoader(
            ImageDataset(opt.test_dir_path, train_opt, imsize=imsize, mode='test', out_channels=out_channels),
            batch_size=train_opt.batch_size,
            shuffle=False,
            num_workers=train_opt.n_cpu,
        )

        print("*****")
        print("Test with : " + checkpoint_name)

        for i, batch in enumerate(dataloader):
            input_mask = Variable(batch["data"].type(Tensor))
            output_img = Variable(batch["gt"].type(Tensor))

            # plt.imshow(output_img[0, 0].cpu())
            # plt.show()

            with torch.no_grad():
                fake_gen = generator(input_mask).cpu().numpy()
                # fake_gen = generator(output_img[0:1, 0:1]).cpu().numpy()

            if out_channels == 1:
                gen_img = np.transpose(fake_gen[0, 0], axes=[0, 1])
            else:
                gen_img = np.transpose(fake_gen[0], axes=[1, 2, 0])

            fake_denormalize = gen_img * 255
            gen_img = Image.fromarray(fake_denormalize.astype(np.uint8))
            gen_img.save("%s/%s/%s/%s/%s" % (opt.checkpoint_model_dir, opt.name, 'outputs', checkpoint_name, os.path.basename(batch["path"][0])))

            print(batch["path"][0] + " saved successfully")

        # for filename_all in filenames_all:
            # Prepare input
            # img = Image.open(filename_all).convert('RGBA')
            # img_data_ratio = ImageDataset.resize_keep_ratio(img, img, imsize)
            # img_data_pad = ImageDataset.pad_image(img_data_ratio, img_data_ratio, imsize)
            # img_data_ratio_pad = img_data_pad(img_data_ratio)
            # tensor_data = Variable(img_data_ratio_pad).to(device).unsqueeze(0)[:, -1:]

            # with torch.no_grad():
            #     fake_gen = generator(tensor_data).cpu().numpy()

            # fake = fake_gen[0, 0:-1]
            # fake = fake_gen[0]
            # fake_ = np.transpose(fake_gen[0], axes=[1, 2, 0])
            # fake_ = np.transpose(fake_gen[0, 0], axes=[0, 1])
            # fake_denormalize = fake_ * 255
            # gen_img = Image.fromarray(fake_denormalize.astype(np.uint8))
            # gen_img.save("%s/%s/%s" % (opt.output_dir, checkpoint_name, os.path.basename(filename_all)))

            # print(filename_all + " saved successfully")
