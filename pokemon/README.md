# Generate Pokemon with DCGAN

*This project will use one variant of Generative Adversarial Network (GAN) known as Deep Convolution GAN or DCGAN to generate new Pokemon. The dataset of Pokemon images come from various sources and preprocessed before inputted to the DCGAN Network. The DCGAN code is derived from [@carpedm20](https://github.com/carpedm20/DCGAN-tensorflow) with some modification.*

## Generative Adversarial Network GAN
GAN consist of two network:

 - A discriminator D receive input from training data and generated data. Its job is to learn how to distinguish between these two inputs.
 - A generator G generate samples from a random noise Z. Generator objective is to generate sample that is as real as possible it could not be distinguished by Discriminator.

 ### Deep Convolution GAN (DCGAN)
In DCGAN architecture, the discriminator D is Convolutional Neural Networks (CNN) that applies a lot of filters to extract various features from an image. The discriminator network will be trained to discriminate between the original and generated image. The process of convolution is shown in the illustration below:

![](http://deeplearning.net/software/theano_versions/dev/_images/same_padding_no_strides_transposed.gif)

The network structure for the discriminator is given by:

| Layer        | Shape           | Activation           |
| ------------- |:-------------:|:-------------:|
| input     | batch size, 3, 64, 64 | |
| convolution      | batch size, 64, 32, 32  | LRelu |
| convolution      | batch size, 128, 16, 16  |LRelu | 
| convolution      | batch size, 256, 8, 8  | LRelu |
| convolution      | batch size, 512, 4, 4 | LRelu |
| dense      | batch size, 64, 32, 32 | Sigmoid |


The generator g, which is trained to generate image to fool the discriminator, is trained to generate image from a random input. In DCGAN architecture, the generator is represented by convolution networks that upsample the input. The goal is to process the small input and make an output that is bigger than the input. It works by expanding the input to have zero in-between and then do the convolution process over this expanded area. The convolution over this area will result in larger input for the next layer. The process of upsampling is shown below: 

![](http://deeplearning.net/software/theano_versions/dev/_images/padding_strides_transposed.gif)

There are many name for this upsample process: full convolution, in-network upsampling, fractionally-strided convolution, deconvolution, or transposed convolution. 

The network structure for the generator is given by:

| Layer        | Shape           | Activation           |
| ------------- |:-------------:|:-------------:|
| input     | batch size, 100 (Noise from uniform distribution) | |
| reshape layer      | batch size, 100, 1, 1  | Relu |
| deconvolution      | batch size, 512, 4, 4   |Relu | 
| deconvolution      | batch size, 256, 8, 8  | Relu |
| deconvolution      | batch size, 128, 16, 16 | Relu |
| deconvolution      | batch size, 64, 32, 32 | Relu |
| deconvolution      | batch size, 3, 64, 64 | Tanh |


 ### Hyperparameter of DCGAN
The hyperparameter for DCGAN architecture is given in the table below:

| Hyperparameter        |
| ------------- |
| Mini-batch size of 64     |
| Weight initialize from normal distribution with std = 0.02      |  
| LRelu slope = 0.2      |
| Adam Optimizer with learning rate = 0.0002 and momentum = 0.5      |


## Pokemon Image Dataset
The dataset of pokemon images are gathered from various sources :
 - https://www.kaggle.com/dollarakshay/pokemon-images/discussion
 - https://veekun.com/dex/downloads

All images will be reshaped to 64x64 pixels with white background. If an image is in png format and has a transparent background (i.e. RGBA), it will be converted to jpg format with RGB channel.

Since there is a limited number of unique Pokemon (around 800), some augmentation technique will be used to generate more training dataset. First, all the image will be flip horizontally. Then all images(original and flipped) is rotated 3, 5, and 7 degrees clockwise and counterclockwise. The training set will be the combination of original, flipped, and rotated images.

# Experiment Result
Here is the training process on unaugmented data

![](https://media.giphy.com/media/3o751ZJJiwArkl9OZG/giphy.gif)

Now I tried to double the dataset by flip it horizontally

![](https://media.giphy.com/media/xULW8sv6Lci0to18oU/giphy.gif)

Finnaly, rotate the image and combine it with flip and original image

![](https://media.giphy.com/media/3oFzmhJedokWQEGiY0/giphy.gif)

From this experiment I observed that Discriminator quickly learn to distinguish between real and fake sample. So I decide to update generator once more whenever the loss between two is bigger than 3.
![](https://preview.ibb.co/j0WyfR/Screenshot_from_2017_12_15_16_44_57.png)

It seems to help. Now I tried to update generator when the loss difference is bigger than 1.
![](https://preview.ibb.co/jQNAum/Screenshot_from_2017_12_15_16_49_22.png)

Next I tried to change the activation function in Generator from ReLU to Leaky ReLU and sample the noise z from normal distribution instead from uniform distribution.
![](https://media.giphy.com/media/xT0xeEvnEgaaxxRrnW/giphy.gif)

Unfortunately, no notable difference in term of image quality and loss function
![](https://preview.ibb.co/cT2Vem/Screenshot_from_2017_12_15_16_49_23.png)

## Pokemon Candidate from This Experiment
![](https://image.ibb.co/bNRqFR/pokemon_candidate.png)

## Run the training process
```sh
python main.py --dataset pokemon --train
```

## Generate from pre-trained model
```sh
python main --dataset pokemon
```









