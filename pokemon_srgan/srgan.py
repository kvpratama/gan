import os
import argparse

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch

import pandas as pd

import matplotlib.pyplot as plt
import pdb

if __name__ == '__main__':

    os.makedirs("saved_models", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
    opt = parser.parse_args()
    print(opt)

    cuda = torch.cuda.is_available()

    hr_shape = (opt.hr_height, opt.hr_width)

    # Initialize generator and discriminator
    generator = GeneratorResNet()
    discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
    feature_extractor = FeatureExtractor()

    # Set feature extractor to inference mode
    feature_extractor.eval()

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_content = torch.nn.L1Loss()

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        feature_extractor = feature_extractor.cuda()
        criterion_GAN = criterion_GAN.cuda()
        criterion_content = criterion_content.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load("saved_models/generator_%d.pth"))
        discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"))

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    dataloader = DataLoader(
        ImageDataset("./data/train/", hr_shape=hr_shape),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    testloader = DataLoader(
        ImageDataset("./data/test/", hr_shape=hr_shape),
        batch_size=1,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    # ----------
    #  Training
    # ----------
    log_list = []

    for epoch in range(opt.epoch, opt.n_epochs):
        total_loss_D, total_loss_G, total_loss_pixel_total, total_loss_content_total, total_loss_GAN_total = 0, 0, 0, 0, 0
        for i, imgs in enumerate(dataloader):

            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["hr"].type(Tensor))

            # plt.subplot(1, 2, 1)
            # plt.imshow(np.transpose(imgs_lr.cpu().numpy()[0], axes=[1, 2, 0]))
            # plt.subplot(1, 2, 2)
            # plt.imshow(np.transpose(imgs_hr.cpu().numpy()[0], axes=[1, 2, 0]))
            # plt.show()
            # pdb.set_trace()

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)

            # Adversarial loss
            loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr)
            loss_content = criterion_content(gen_features, real_features.detach())

            # Total loss
            loss_G = loss_content + 1e-3 * loss_GAN

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss of real and fake images
            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                  % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item()))

            total_loss_D += loss_D.item()
            total_loss_G += loss_G.item()
            total_loss_content_total += loss_content.item()
            total_loss_GAN_total += loss_GAN.item()

        log_list.append((epoch, total_loss_D / len(dataloader), total_loss_G / len(dataloader),
                         total_loss_pixel_total / len(dataloader), total_loss_content_total / len(dataloader),
                         total_loss_GAN_total / len(dataloader)))
        log_df = pd.DataFrame(log_list, columns=['epoch', 'loss_d', 'loss_g', 'loss_pixel', 'loss_content', 'loss_GAN'])
        log_df.to_csv("log.csv")

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "./saved_models/generator_%d.pth" % epoch)
            torch.save(discriminator.state_dict(), "./saved_models/discriminator_%d.pth" % epoch)
            plt.figure(figsize=(25, 9))

            for j, imgs in enumerate(testloader):
                imgs_lr = Variable(imgs["lr"].type(Tensor))
                imgs_hr = Variable(imgs["hr"].type(Tensor))

                with torch.no_grad():
                    gen_hr = generator(imgs_lr)

                imgs_lr = np.transpose(imgs_lr.cpu().numpy()[0], axes=[1, 2, 0])
                imgs_hr = np.transpose(imgs_hr.cpu().numpy()[0], axes=[1, 2, 0])
                gen_hr = np.transpose(gen_hr.detach().cpu().numpy()[0], axes=[1, 2, 0])

                imgs_lr_denorm = imgs_lr * 0.5 + 0.5
                imgs_hr_denorm = imgs_hr * 0.5 + 0.5
                gen_hr_denorm = gen_hr * 0.5 + 0.5

                plt.subplot(9, 3, j*3+1)
                plt.imshow(imgs_lr_denorm)
                plt.subplot(9, 3, j*3+2)
                plt.imshow(imgs_hr_denorm)
                plt.subplot(9, 3, j*3+3)
                plt.imshow(gen_hr_denorm)

            plt.savefig("./saved_models/%d.jpg" % epoch)
            # pdb.set_trace()

