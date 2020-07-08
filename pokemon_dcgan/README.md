# DCGAN with Differentiable Augmentation in Pytorch

Differentiable Augmentation (*DiffAugment*) is a technique to train GAN that was published in June 2020.
Simply put, *DiffAugment* is a Data Augmentation technique design specifically for GAN. 
It addresses the problem when the Discriminator memorizes the training set due to a limited amount of data.
Traditionally, we simply add more data to the training set. 
But most of the time collecting more data is expensive and time-consuming.

This project explores the effect of using *DiffAugment* in *DCGAN* model to create a new Pokemon. 
The dataset being used only contains 819 Pokemon images which are perfect for experimenting and seeing the effect of *DiffAugment*.
You can found the dataset in this [link](https://www.kaggle.com/kvpratama/pokemon-images-dataset).

## Run the training

To train the DCGAN network simply run this command:
`python train.py --name "pokemon_diffaug03" --aug_prob 0.3`

`--name` specifies the experiment name, `--aug_prob` specify the probability of using *DiffAugment* during training.
You can enter a float value ranging from 0, which means no augmentation, to 1, which means use *DiffAugment* in every iteration.


## Training Progression
Below is the generated gif showing examples of generated images during training.

|DCGAN             |DCGAN + *DiffAugment* 0.3         |DCGAN + *DiffAugment* 0.5     |DCGAN + *DiffAugment* 1
|---               |---                          |---                           |---   
|<img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/progress.gif" width="256">|<img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/progress_diffaug0.3.gif" width="256">|<img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/progress_diffaug0.5.gif" width="256">|<img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/progress_diffaug1.gif" width="256">|

As you might notice, DCGAN without *DiffAugment* starts to generate random noise. This happens to start at epoch 226. 
The loss graph below could be a hint of what happened during training. 

<img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/loss.jpg" width="512">

Even though it managed to escape this mode around epoch 875, the image quality is unsatisfactory. 
Furthermore, on the last epoch, this model starts showing the tendency to generate random noise again as shown in the figure below.

|DCGAN             |DCGAN + *DiffAugment* 0.3         |DCGAN + *DiffAugment* 0.5     |DCGAN + *DiffAugment* 1
|---               |---                          |---                           |---   
|<img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/1000_dcgan.jpg" width="256">|<img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/1000_dcgan_03.jpg" width="256">|<img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/1000_dcgan_05.jpg" width="256">|<img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/1000_dcgan_1.jpg" width="256">|

From the figure above, we can also compare the effect of *DiffAugment* with a different probability.
DCGAN + *DiffAugment* with a probability of 0.3 tends to generate a somewhat similar-looking Pokemon. 
This was alleviated in DCGAN + *DiffAugment* with a probability of 0.5. 
DCGAN that use *DiffAugment* in every iteration produces the most varying Pokemon in term of color and shape.
It is safe to conclude that *DiffAugment* helps Generator learn to generate more diverse samples.


## FID Score across epoch
Finally, let's look at the FID Score in the graph below.

<img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/fid.jpg" width="512">

Training without *DiffAugment* pushes the network into Mode Collapse where G just generates a random noise with a very high FID score.
On the other hand, *DiffAugment* helps stabilized the FID score, and the higher the probability the less fluctuation in FID score observed.


## Some Pokemon candidate from this experiment:

<img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/9.jpeg"><img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/57.jpeg"><img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/58.jpeg"><img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/59.jpeg"><img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/225.jpeg"><img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/278.jpeg"><img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/279.jpeg"><img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/449.jpeg"><img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/590.jpeg"><img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/641.jpeg"><img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/777.jpeg"><img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/939.jpeg"><img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/993.jpeg">


DCGAN code is derived from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

*DiffAugment* code is derived from https://github.com/mit-han-lab/data-efficient-gans/blob/master/DiffAugment_pytorch.py

