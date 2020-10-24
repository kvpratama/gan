import argparse
import os
import random
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from model import *
from cyclegan import *
import itertools
from DiffAugmentation import DiffAugment
import pdb


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def sample_images(opt, batches_done, monet_dataloader, photo_dataloader):
    """Saves a generated sample from the test set"""
    G_AB.eval()
    G_BA.eval()
    real_A = next(iter(monet_dataloader))[0].cuda()
    fake_B = G_AB(real_A)
    real_B = next(iter(photo_dataloader))[0].cuda()
    fake_A = G_BA(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=opt.batch_size, normalize=True)
    real_B = make_grid(real_B, nrow=opt.batch_size, normalize=True)
    fake_A = make_grid(fake_A, nrow=opt.batch_size, normalize=True)
    fake_B = make_grid(fake_B, nrow=opt.batch_size, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "%s/%s/images/%s.png" % (opt.checkpoints_dir, opt.name, batches_done), normalize=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="./data", help="Root directory for dataset")
    parser.add_argument("--name", type=str, default="monet", help="Checkpoint name")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--workers", type=int, default=2, help="Number of workers for dataloader")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size during training")
    parser.add_argument("--image_size", type=int, default=264, help="spatial size of training images. All images will be resized to this size")
    parser.add_argument("--nc", type=int, default=3, help="Number of channels in the training images. For color images this is 3")
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="checkpoint save interval")
    parser.add_argument("--sample_interval", type=int, default=100, help="save sample output interval")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
    parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
    parser.add_argument("--aug_prob", type=float, default=0, help="probability of using Diff Augmentation during training")
    opt = parser.parse_args()
    print(opt)

    # Set random seed for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    os.makedirs("%s/%s/images" % (opt.checkpoints_dir, opt.name), exist_ok=True)
    os.makedirs("%s/%s/saved_models" % (opt.checkpoints_dir, opt.name), exist_ok=True)

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Create the dataset
    monet_dataset = dset.ImageFolder(root=opt.dataroot+'/monet_jpg',
                               transform=transforms.Compose([
                                   transforms.Resize(opt.image_size),
                                   transforms.CenterCrop(opt.image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    photo_dataset = dset.ImageFolder(root=opt.dataroot + '/photo_jpg',
                                     transform=transforms.Compose([
                                         transforms.Resize(opt.image_size),
                                         transforms.CenterCrop(opt.image_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     ]))
    # Create the dataloader
    monet_dataloader = torch.utils.data.DataLoader(monet_dataset, batch_size=opt.batch_size,
                                             shuffle=True, num_workers=opt.workers)
    photo_dataloader = torch.utils.data.DataLoader(photo_dataset, batch_size=opt.batch_size,
                                                   shuffle=True, num_workers=opt.workers)

    # Plot some training images
    # real_batch = next(iter(photo_dataloader))
    # plt.figure(figsize=(8,8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    # plt.show()

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    input_shape = (opt.nc, opt.image_size, opt.image_size)

    # Initialize generator and discriminator
    G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
    G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
    D_A = Discriminator(input_shape)
    D_B = Discriminator(input_shape)

    if cuda:
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
        D_A = D_A.cuda()
        D_B = D_B.cuda()
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )

    # Buffers of previously generated samples
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # ----------
    #  Training
    # ----------

    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(monet_dataloader):
            rand_prob = np.random.uniform(0, 1)

            # Set model input
            real_A = batch[0].cuda()
            real_B = next(iter(photo_dataloader))[0].cuda()
            # pdb.set_trace()
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            optimizer_D_A.zero_grad()

            # if rand_prob < opt.aug_prob:
            #     real_A = DiffAugment(real_A, policy='color,translation,cutout')
            #     fake_A = DiffAugment(fake_A.detach(), policy='color,translation,cutout')
            # # Real loss
            # loss_real = criterion_GAN(D_A(real_A), valid)
            # # Fake loss
            # loss_fake = criterion_GAN(D_A(fake_A), fake)
            # else:
            # Real loss
            if rand_prob < opt.aug_prob:
                loss_real = criterion_GAN(D_A(DiffAugment(real_A, policy='color,translation,cutout')), valid)
                fake_A_ = fake_A_buffer.push_and_pop(DiffAugment(fake_A, policy='color,translation,cutout'))
            else:
                loss_real = criterion_GAN(D_A(real_A), valid)
                fake_A_ = fake_A_buffer.push_and_pop(fake_A)

            loss_fake = criterion_GAN(D_A(fake_A_), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            optimizer_D_B.zero_grad()

            # if rand_prob < opt.aug_prob:
            #     real_B = DiffAugment(real_B, policy='color,translation,cutout')
            #     fake_B = DiffAugment(fake_B, policy='color,translation,cutout')
            # # Real loss
            # loss_real = criterion_GAN(D_A(real_B), valid)
            # # Fake loss
            # loss_fake = criterion_GAN(D_A(fake_B), fake)
            # else:
            # Real loss
            if rand_prob < opt.aug_prob:
                loss_real = criterion_GAN(D_B(DiffAugment(real_B, policy='color,translation,cutout')), valid)
                fake_B_ = fake_B_buffer.push_and_pop(DiffAugment(fake_B, policy='color,translation,cutout'))
            else:
                loss_real = criterion_GAN(D_B(real_B), valid)
                fake_B_ = fake_B_buffer.push_and_pop(fake_B)

            loss_fake = criterion_GAN(D_A(fake_B_), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(monet_dataloader) + i
            batches_left = opt.n_epochs * len(monet_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                  % (
                      epoch,
                      opt.n_epochs,
                      i,
                      len(monet_dataloader),
                      loss_D.item(),
                      loss_G.item(),
                      loss_GAN.item(),
                      loss_cycle.item(),
                      loss_identity.item(),
                      time_left,
                  ))

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_images(opt, batches_done, monet_dataloader, photo_dataloader)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), "%s/%s/saved_models/G_AB_%d.pth" % (opt.checkpoints_dir, opt.name, epoch))
            torch.save(G_BA.state_dict(), "%s/%s/saved_models/G_BA_%d.pth" % (opt.checkpoints_dir, opt.name, epoch))
            torch.save(D_A.state_dict(), "%s/%s/saved_models/D_A_%d.pth" % (opt.checkpoints_dir, opt.name, epoch))
            torch.save(D_B.state_dict(), "%s/%s/saved_models/D_B_%d.pth" % (opt.checkpoints_dir, opt.name, epoch))

