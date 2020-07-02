import argparse
import os
import torch
import random
import glob

from model import *

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import pdb

'''Generate images from saved checkpoints'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="pokemon2", help="Root directory for dataset")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Root directory for dataset")
    parser.add_argument("--n_image", type=int, default=1000, help="Number of image to generate")
    opt = parser.parse_args()
    print(opt)

    # Set random seed for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = torch.device("cuda:0")

    fixed_noise = torch.randn(opt.n_image, 100, 1, 1, device=device)

    g_checkpoints = glob.glob(os.path.join(opt.checkpoints_dir, opt.name, "saved_models", "generator*"))

    for checkpoint in g_checkpoints:
        gen_state = torch.load(checkpoint)
        opt = gen_state['opt']
        netG = Generator(opt).to(device)
        netG.load_state_dict(gen_state.get('weight', False))

        netG.eval()
        savedir = "%s/%s/generated/%s" % (opt.checkpoints_dir, opt.name, os.path.basename(checkpoint).split('.')[0])
        os.makedirs(savedir, exist_ok=True)

        with torch.no_grad():
            for i in range(fixed_noise.size(0)):
                fakegen = netG(fixed_noise[i:i+1])
                fake = (fakegen * 0.5 + 0.5) * 255
                im = Image.fromarray(np.transpose(fake[0].cpu().numpy().astype(np.uint8), axes=[1, 2, 0]))
                im.save(os.path.join(savedir, str(i) + ".jpeg"))

        print(savedir, 'finished')

