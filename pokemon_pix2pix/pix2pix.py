import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
    parser.add_argument("--name", type=str, default="pokemon", help="name of the dataset")
    parser.add_argument('--data_dir', type=str, default='.\\data')
    parser.add_argument('--checkpoints_dir', type=str, default='.\\checkpoints', help='models are saved here')
    parser.add_argument('--save_freq', type=int, default=10, help='Frequency of saving model per epoch')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--lambda_pixel", type=int, default=0,
                        help="Loss weight of L1 pixel-wise loss between translated image and real image")
    parser.add_argument("--lambda_content", type=int, default=0,
                        help="Loss weight of vgg loss between translated image and real image")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--imsize", type=int, default=256, help="size of image height and width")
    parser.add_argument("--in_channels", type=int, default=1, help="number of input image channels")
    parser.add_argument("--out_channels", type=int, default=3, help="number of output image channels")
    parser.add_argument("--edges_to_sketch", action='store_true', help="Translation from mask to sketch")
    parser.add_argument("--sketch_to_color", action='store_true', help="Translation from sketch color image")
    parser.add_argument("--reverse", action='store_true', help="Reverse the translation task")
    parser.add_argument("--l1_crop", action='store_true', help="Use cropped area for L1 loss (do not use with augmentation)")
    parser.add_argument("--augmentation", action='store_true', help="Use data augmentation technique (do not use with li_crop)")
    parser.add_argument("--sigma", type=int, default=1, help="parameter sigma that control sketch strength")
    parser.add_argument("--threshold", type=int, default=250, help="threshold for skethcify")

    # parser.add_argument('--resize_input', action='store_true', help='Resize and padding the input to --imsize')
    opt = parser.parse_args()
    print(opt)

    os.makedirs("%s/%s/images" % (opt.checkpoints_dir, opt.name), exist_ok=True)
    os.makedirs("%s/%s/saved_models" % (opt.checkpoints_dir, opt.name), exist_ok=True)

    cuda = True if torch.cuda.is_available() else False

    device_count = torch.cuda.device_count()

    cuda0 = torch.device('cuda:0')

    if device_count > 1:
        cuda1 = torch.device('cuda:1')
    else:
        cuda1 = torch.device('cuda:0')

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    criterion_content = torch.nn.L1Loss()

    # Loss weight of L1 pixel-wise loss between translated image and real image
    lambda_pixel = opt.lambda_pixel
    lambda_content = opt.lambda_content

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, opt.imsize // 2 ** 4, opt.imsize // 2 ** 4)

    # Initialize generator and discriminator
    generator = GeneratorUNet(in_channels=opt.in_channels, out_channels=opt.out_channels)
    discriminator = Discriminator(in_channels=opt.out_channels + opt.in_channels)

    feature_extractor = FeatureExtractor()
    # Set feature extractor to inference mode
    feature_extractor.eval()

    if cuda:
        generator = generator.cuda(cuda1)
        discriminator = discriminator.cuda(cuda0)
        feature_extractor = feature_extractor.cuda(cuda0)
        criterion_GAN.cuda(cuda0)
        criterion_pixelwise.cuda(cuda0)
        criterion_content.cuda(cuda0)

    if opt.epoch != 0:
        # Load pretrained models
        gen_state = torch.load("%s/%s/saved_models/generator_%d.pth" % (opt.checkpoints_dir, opt.name, opt.epoch),
                               map_location=cuda1)
        # disc_state = torch.load("%s/%s/saved_models/discriminator_%d.pth" % (opt.checkpoints_dir, opt.name, opt.epoch), map_location=cuda1)
        # opt = gen_state['opt']
        generator.load_state_dict(gen_state.get('weight', False))
        discriminator.load_state_dict(
            torch.load("%s/%s/saved_models/discriminator_%d.pth" % (opt.checkpoints_dir, opt.name, opt.epoch),
                       map_location=cuda1))
        print('Pretrained model ', opt.epoch, ' loaded')
    else:
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)
        # Save parameter to csv file
        with open('%s/%s/parameters.csv' % (opt.checkpoints_dir, opt.name), 'w') as f:
            for key, value in vars(opt).items():
                f.write("%s,%s\n" % (key, value))

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    dataloader = DataLoader(
        ImageDataset(opt.data_dir, opt, imsize=opt.imsize, mode=opt.phase, out_channels=opt.out_channels),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    dataloader_test = DataLoader(
        ImageDataset(opt.data_dir, opt, imsize=opt.imsize, mode='test', out_channels=opt.out_channels),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    prev_time = time.time()
    log_list = []

    for epoch in range(opt.epoch, opt.n_epochs):
        total_loss_D, total_loss_G, total_loss_pixel_total, total_loss_content_total, total_loss_GAN_total = 0, 0, 0, 0, 0
        for i, batch in enumerate(dataloader):

            # Model inputs
            input_mask = Variable(batch["data"].type(Tensor))
            output_img = Variable(batch["gt"].type(Tensor))

            # plt.subplot(1, 2, 1)
            # plt.imshow(input_mask.cpu()[0, 0], cmap='gray')
            # plt.imshow(np.transpose(input_mask.cpu()[0, :3], axes=[1, 2, 0]))
            # plt.subplot(1, 2, 2)
            # plt.imshow(np.transpose(output_img[0, :3].cpu(), axes=[1, 2, 0]))
            # plt.imshow(output_img[0, 0].cpu(), cmap='gray')
            # plt.imshow(input_mask.cpu()[0, 0,
            #            batch['mask_index'][0]:batch['mask_index'][1],
            #            batch['mask_index'][2]:batch['mask_index'][3]], cmap='gray')
            # plt.show()
            # pdb.set_trace()

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((input_mask.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((input_mask.size(0), *patch))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # GAN loss
            fake_gen = generator(input_mask.cuda(cuda1))

            pred_fake = discriminator(fake_gen.cuda(cuda0), input_mask)
            loss_GAN = criterion_GAN(pred_fake, valid)

            if opt.l1_crop:
                row_min = batch['mask_index'][0]
                row_max = batch['mask_index'][1]
                col_min = batch['mask_index'][2]
                col_max = batch['mask_index'][3]
                fake_gen_ = fake_gen[:, :, row_min:row_max, col_min:col_max]
                output_img_ = output_img[:, :, row_min:row_max, col_min:col_max]
            else:
                fake_gen_ = fake_gen
                output_img_ = output_img
            # plt.subplot(1, 2, 1)
            # plt.imshow(fake_gen_.cpu().detach()[0, 0], cmap='gray')
            # plt.subplot(1, 2, 2)
            # plt.imshow(output_img_.cpu().detach()[0, 0], cmap='gray')
            # plt.show()
            loss_pixel = criterion_pixelwise(fake_gen_.cuda(cuda0), output_img_)
            gen_features = feature_extractor(fake_gen_.cuda(cuda0))
            real_features = feature_extractor(output_img_)
            loss_content = criterion_content(gen_features, real_features.detach())

            # Total loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel + lambda_content * loss_content
            # loss_G = loss_GAN1 + loss_GAN2 + (lambda_pixel * loss_pixel1) + (lambda_pixel * loss_pixel2
            # loss_G = loss_GAN_total + loss_pixel_total

            loss_G.backward()

            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(output_img, input_mask)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_gen.cuda(cuda0).detach(), input_mask)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)
            # loss_D = 0.5 * (loss_real1 + loss_real2 + loss_fake1 + loss_fake2)
            # loss_D = 0.5 * (loss_real_total + loss_fake_total)

            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, content: %f, adv: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_pixel,
                    loss_content,
                    loss_GAN,
                    time_left,
                )
            )
            total_loss_D += loss_D.item()
            total_loss_G += loss_G.item()
            total_loss_pixel_total += loss_pixel.item()
            total_loss_content_total += loss_content.item()
            total_loss_GAN_total += loss_GAN.item()

        log_list.append((epoch, total_loss_D / len(dataloader), total_loss_G / len(dataloader),
                         total_loss_pixel_total / len(dataloader), total_loss_content_total / len(dataloader),
                         total_loss_GAN_total / len(dataloader)))

        for i, batch in enumerate(dataloader_test):
            input_mask = Variable(batch["data"].type(Tensor))
            output_img = Variable(batch["gt"].type(Tensor))

            # plt.imshow(output_img[0, 0].cpu())
            # plt.show()

            with torch.no_grad():
                fake_gen = generator(input_mask)

            if opt.out_channels == 1:
                gen_img = np.transpose(fake_gen.cpu().detach().numpy()[0, 0], axes=[0, 1])
                out_img = np.transpose(output_img.cpu().detach().numpy()[0, 0], axes=[0, 1])
            else:
                gen_img = np.transpose(fake_gen.cpu().detach().numpy()[0], axes=[1, 2, 0])
                out_img = np.transpose(output_img.cpu().detach().numpy()[0], axes=[1, 2, 0])

            gen_img_denormalize = gen_img * 255
            # img = Image.fromarray(gen_img_denormalize.astype(np.uint8), 'RGB')
            img = Image.fromarray(gen_img_denormalize.astype(np.uint8))
            img.save("%s/%s/images/%d_gen_%d.png" % (opt.checkpoints_dir, opt.name, epoch, i))

            # out_img_denormalize = out_img * 255
            # img = Image.fromarray(out_img_denormalize.astype(np.uint8))
            # img.save("%s/%s/images/%d_target.png" % (opt.checkpoints_dir, opt.name, epoch))

        if epoch % opt.save_freq == 0 or epoch == opt.n_epochs - 1:
            state = {'epoch_done': epoch, 'min_val_data': dataloader.dataset.data_min_val,
                     'max_val_data': dataloader.dataset.data_max_val,
                     'opt': opt, 'weight': generator.state_dict()}
            torch.save(state, "%s/%s/saved_models/generator_%d.pth" % (opt.checkpoints_dir, opt.name, epoch))
            torch.save(discriminator.state_dict(),
                       "%s/%s/saved_models/discriminator_%d.pth" % (opt.checkpoints_dir, opt.name, epoch))

        log_df = pd.DataFrame(log_list, columns=['epoch', 'loss_d', 'loss_g', 'loss_pixel', 'loss_content', 'loss_GAN'])
        log_df.to_csv("%s/%s/log.csv" % (opt.checkpoints_dir, opt.name))
