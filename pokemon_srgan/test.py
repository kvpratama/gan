import argparse
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models import *
from datasets import *

import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    opt = parser.parse_args()
    print(opt)

    hr_shape = (opt.hr_height, opt.hr_width)

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    generator = GeneratorResNet()

    if cuda:
        generator = generator.cuda()

    generator.load_state_dict(torch.load("saved_models/generator_99.pth"))

    testloader = DataLoader(
        ImageDataset("./data/test/", hr_shape=hr_shape),
        batch_size=1,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    for i, imgs in enumerate(testloader):
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        with torch.no_grad():
            gen_hr = generator(imgs_lr)

        imgs_lr = np.transpose(imgs_lr.detach().cpu().numpy()[0], axes=[1, 2, 0])
        imgs_hr = np.transpose(imgs_hr.detach().cpu().numpy()[0], axes=[1, 2, 0])
        gen_hr = np.transpose(gen_hr.detach().cpu().numpy()[0], axes=[1, 2, 0])

        imgs_lr_denorm = imgs_lr * 0.5 + 0.5
        imgs_hr_denorm = imgs_hr * 0.5 + 0.5
        gen_hr_denorm = gen_hr * 0.5 + 0.5

        ax = plt.subplot(1, 3, 1)
        ax.set_title('Low Resolution')
        ax.imshow(imgs_lr_denorm)
        ax = plt.subplot(1, 3, 2)
        ax.set_title('High Resolution')
        ax.imshow(imgs_hr_denorm)
        ax = plt.subplot(1, 3, 3)
        ax.set_title('Super Resolution')
        plt.imshow(gen_hr_denorm)
        plt.savefig("./saved_models/test_%d.jpg" % i)
        # plt.show()
