import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from model import *
from DiffAugmentation import DiffAugment

import pdb

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="./data", help="Root directory for dataset")
    parser.add_argument("--name", type=str, default="pokemon", help="Root directory for dataset")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Root directory for dataset")
    parser.add_argument("--workers", type=int, default=2, help="Number of workers for dataloader")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size during training")
    parser.add_argument("--image_size", type=int, default=64, help="spatial size of training images. All images will be resized to this size")
    parser.add_argument("--nc", type=int, default=3, help="Number of channels in the training images. For color images this is 3")
    parser.add_argument("--nz", type=int, default=100, help="Size of z latent vector (i.e. size of generator input)")
    parser.add_argument("--ngf", type=int, default=64, help="Size of feature maps in generator")
    parser.add_argument("--ndf", type=int, default=64, help="Size of feature maps in discriminator")
    parser.add_argument("--num_epochs", type=int, default=50, help="Size of feature maps in discriminator")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
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

    ngpu = torch.cuda.device_count()

    # Create the dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                                transform=transforms.Compose([
                                    transforms.Resize(opt.image_size),
                                    transforms.CenterCrop(opt.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=opt.workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Plot some training images
    # real_batch = next(iter(dataloader))
    # plt.figure(figsize=(8,8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    # plt.show()

    netG = Generator(opt).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Print the G
    print(netG)

    netD = Discriminator(opt).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(64, opt.nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    # iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(opt.num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            rand_prob = np.random.uniform(0, 1)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            if rand_prob < opt.aug_prob:
                real_cpu = DiffAugment(real_cpu, policy='color,translation,cutout')
                # plt.subplot(1, 2, 1)
                # plt.imshow(np.transpose(data[0][1].cpu().numpy(), axes=[1, 2, 0]))
                # plt.subplot(1, 2, 2)
                # plt.imshow(np.transpose(aug_data[1].cpu().numpy(), axes=[1, 2, 0]))
                # plt.show()
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, opt.nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            if rand_prob < opt.aug_prob:
                fake = DiffAugment(fake, policy='color,translation,cutout')
                # plt.subplot(1, 2, 1)
                # plt.imshow(np.transpose(fake[0].cpu().detach().numpy(), axes=[1, 2, 0]))
                # plt.subplot(1, 2, 2)
                # plt.imshow(np.transpose(fake_aug[0].cpu().detach().numpy(), axes=[1, 2, 0]))
                # plt.show()

            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch+1, opt.num_epochs, i+1, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        if epoch % 5 == 0 or epoch == opt.num_epochs - 1:
            state = {'epoch_done': epoch, 'opt': opt, 'weight': netG.state_dict()}
            torch.save(state, "%s/%s/saved_models/generator_%d.pth" % (opt.checkpoints_dir, opt.name, epoch))
            torch.save(netD.state_dict(),
                       "%s/%s/saved_models/discriminator_%d.pth" % (opt.checkpoints_dir, opt.name, epoch))

            # Check how the generator is doing by saving G's output on fixed_noise
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            vutils.save_image(fake, "%s/%s/images/%d.jpg" % (opt.checkpoints_dir, opt.name, epoch))


    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(opt.checkpoints_dir, opt.name, 'loss.jpg'))
    plt.clf()

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    ani.save(os.path.join(opt.checkpoints_dir, opt.name, 'progress.gif'))
