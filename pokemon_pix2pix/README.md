# Pix2Pix Image to Image translation

Link to the Pokemon images dataset
https://www.kaggle.com/kvpratama/pokemon-images-dataset

Code are derive from 
https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/pix2pix

Run the training for color to sketch translation:

`python pix2pix.py --name color_to_sketch --sketch_to_color --reverse --sigma 1 --in_channels 3 --out_channels 1 --lambda_pixel 100`

Run the training for sketch to color translation:

`python pix2pix.py --name sketch_to_color --sketch_to_color --out_channels 3 --lambda_pixel 100 --augmentation`

I use the code from this [blog](https://www.freecodecamp.org/news/sketchify-turn-any-image-into-a-pencil-sketch-with-10-lines-of-code-cf67fa4f68ce/) to create a sketch image.

## Color to Sketch Examples on the test set

![](https://github.com/kvpratama/gan/blob/master/pokemon_pix2pix/assets/563.png)
<img src="https://github.com/kvpratama/gan/blob/master/pokemon_pix2pix/assets/563.gif" width="256">

![](https://github.com/kvpratama/gan/blob/master/pokemon_pix2pix/assets/610.png)
<img src="https://github.com/kvpratama/gan/blob/master/pokemon_pix2pix/assets/610.gif" width="256">

![](https://github.com/kvpratama/gan/blob/master/pokemon_pix2pix/assets/681.png)
<img src="https://github.com/kvpratama/gan/blob/master/pokemon_pix2pix/assets/681.gif" width="256">
