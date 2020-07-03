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
Below is the generated gif showing examples of generated image during training.

|DCGAN   	        |DCGAN + *DiffAugment* 0.3 	    |DCGAN + *DiffAugment* 0.5  	|DCGAN + *DiffAugment* 1
|---	            |---	            	        |---	                        |---	
|<img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/progress.gif" width="256">|<img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/progress_diffaug0.3.gif" width="256">|<img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/progress_diffaug0.5.gif" width="256">|<img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/progress_diffaug1.gif" width="256">|

As you might notice, DCGAN without *DiffAugment* start to generate random noise.This happen starting from epoch 226. 
The loss graph below could be a hint of what happened during training. 

<img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/loss.jpg" width="256">

Even though it managed to escape this mode around epoch 875, the image quality is unsatisfactory. 
Furthermore, on the last epoch, this model start showing the tendency to generate random noise again as shown in the figure below.

|DCGAN   	        |DCGAN + *DiffAugment* 0.3 	    |DCGAN + *DiffAugment* 0.5  	|DCGAN + *DiffAugment* 1
|---	            |---	            	        |---	                        |---	
|<img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/1000_dcgan.jpg" width="256">|<img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/1000_dcgan_03.gif" width="256">|<img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/1000_dcgan_05.gif" width="256">|<img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/1000_dcgan_1" width="256">|

From the figure above, we can also compare the effect of *DiffAugment* with different probability.
DCGAN + *DiffAugment* with probability of 0.3 tend to generate a somewhat similar looking Pokemon. 
This was alleviate in DCGAN + *DiffAugment* with probability of 0.5. 
DCGAN that use *DiffAugment* in every iteration produce the most varying Pokemon in term of color and shape.
It is safe to conclude that using *DiffAugment* help generator learn to generate a more diverse samples.

Some Pokemon candidate from this experiment:
<img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/9.jpg"><img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/57.jpg"><img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/58.jpg"><img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/59.jpg"><img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/225.jpg"><img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/278.jpg"><img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/279.jpg"><img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/449.jpg"><img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/590.jpg"><img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/641.jpg"><img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/777.jpg"><img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/939.jpg"><img src="https://github.com/kvpratama/gan/blob/master/pokemon_dcgan/assets/candidates/993.jpg">


DCGAN code is derived from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

*DiffAugment* code is derived from https://github.com/mit-han-lab/data-efficient-gans/blob/master/DiffAugment_pytorch.py

