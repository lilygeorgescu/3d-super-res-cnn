## Convolutional Neural Networks with Intermediate Loss for 3D Super-Resolution of CT and MRI Scans
Code for training CNN for 3D medical images super resolution.

### 1. License agreement
Copyright (C) 2020 Mariana Iuliana Georgescu, Radu Tudor Ionescu

Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) 

(https://creativecommons.org/licenses/by-nc-sa/4.0/)

You are free to:

  #### Share — copy and redistribute the material in any medium or format

  #### Adapt — remix, transform, and build upon the material

Under the following terms:

 #### Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.


 #### NonCommercial — You may not use the material for commercial purposes.


 #### ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.


### 2. Citation

Please cite the following work [1] if you use this software (or a modified version of it) in any scientific
 work:
 
[1] Mariana-Iuliana Georgescu and Radu Tudor Ionescu and Nicolae Verga. Convolutional Neural Networks with Intermediate Loss for 3D Super-Resolution of CT and MRI Scans, IEEE Access 2020
 
Bibtex:
```
@misc{Georgescu-2020,
    title={Convolutional Neural Networks with Intermediate Loss for 3D Super-Resolution of CT and MRI Scans},
    author={Mariana-Iuliana Georgescu and Radu Tudor Ionescu and Nicolae Verga},
    year={2020}, 
    book={IEEE Access}
}
```
### 3. Train your own model

#### 3.1 Prepare the data set.
In order to train your model, you first need to prepare your data (The scripts written in Matlab, extract patches from the images, 
apply a random gaussian filter on the input patch image and save the input patches and the ground-truth ones.).

For training on two axes, use the following scripts:
matlab/h_w/resize_images_for_training.m
matlab/h_w/resize_images_for_testing.m

For training on one axis, use the following scripts:
matlab/d/resize_images_for_training.m
matlab/d/resize_images_for_testing.m

Set the ```folder_name``` and ```resize_factor``` according to your needs.

your dataset should have the following format:

path/to/your/data/set/image_name/input_14_14_2/1.png

path/to/your/data/set/image_name/input_14_14_2/2.png

.....

path/to/your/data/set/image_name/gt_14_14_2/1.png

path/to/your/data/set/image_name/gt_14_14_2/2.png

......


#### 3.2 Train the network
For training on two axes, use ```h and w/train.py```.  

Modify the paths to your data set in the ```DataReader``` object (h and w/train.py line 25)

Set the ```dim_patch``` from ```h and w/parameters.py``` to your ```dim_patch``` set when you prepared the dataset.

If your you did not use the Matlab scripts to generate the dataset modify the paths to your dataset in 
```h and w/data_reader.py line 23, 24```.


For training on one axis, use ```d/train.py```.  

Modify the paths to your data set in the ```DataReader``` object (d/train.py line 25)

Set the ```dim_patch``` from ```d/parameters.py``` to your ```dim_patch``` set when you prepared the dataset.

If your you did not use the Matlab scripts to generate the dataset modify the paths to your dataset in 
```d/data_reader.py line 24, 26```.

#### 3.2 Evaluate the network 

Use the scripts ```d/eval.py``` (one axis) ```h and w/eval.py``` (two axes).

Change the ```test_path``` variable to point to your dataset (maybe you need to modify the ```read_images``` function according to your needs).

In the ```evaluate.py``` script, we flip and rotate the image (90 degrees) and obtain the final SR image by averaging the augmented images.