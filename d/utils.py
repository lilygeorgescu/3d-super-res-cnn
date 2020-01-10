#  Copyright (c) 2020. Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Nicolae Verga
#  Convolutional Neural Networks with  Intermediate Loss for 3D Super-Resolution of CT and MRI Scans
#  Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
#  (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#

import numpy as np
import cv2 as cv
from PIL import Image
import params as params
import os
import glob
import numpy as np
from skimage.measure import compare_ssim as ssim_sk 
from skimage.measure import compare_psnr as psnr_sk 

import math 
import pdb

SHOW_IMAGES = False


def my_psnr(img1, img2):
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0 
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def psnr(img1, img2):  
    img1[img1 < 0] = 0
    img1[img1 > 255] = 255
    
    img2[img2 < 0] = 0
    img2[img2 > 255] = 255
    
    img1 = np.uint8(img1)
    img2 = np.uint8(img2)
    res = psnr_sk(img1, img2)
    if math.isinf(res):
        return 100
    else:
        return res


def ssim(img1, img2):
    img1[img1 < 0] = 0
    img1[img1 > 255] = 255

    img2[img2 < 0] = 0
    img2[img2 > 255] = 255

    img1 = np.uint8(img1)
    img2 = np.uint8(img2)
    if(img1.shape[2] == 1):
        return ssim_sk(np.squeeze(img1), np.squeeze(img2))
    return ssim_sk(img1, img2)


def compute_ssim_psnr_batch(predicted_images, ground_truth_images):
    num_images = predicted_images.shape[0]
    ssim_sum = 0
    psnr_sum = 0 
    for i in range(num_images):
        ssim_sum += ssim(predicted_images[i], ground_truth_images[i])
        psnr_sum += psnr(predicted_images[i], ground_truth_images[i])
        
    return ssim_sum, psnr_sum


def compute_ssim_psnr(predicted_images, original_images, stride=None):
    
    if(stride is not None):
        original_images = original_images[:, stride:-stride, stride:-stride, :].copy()
        
    if(stride is not None and original_images.shape != predicted_images.shape):
        predicted_images = predicted_images[:, stride:-stride, stride:-stride, :].copy()
    
    num_images = original_images.shape[0]
    sum_ssim = 0
    sum_psnr = 0
    for index in range(num_images):  
        psnr_value = psnr(original_images[index], predicted_images[index])
        ssim_value = ssim(original_images[index], predicted_images[index])  
        
        sum_ssim += ssim_value
        sum_psnr += psnr_value 
    
    return sum_ssim / num_images, sum_psnr / num_images


def rotate_image_90(image):
    return np.rot90(image.copy())


def rotate_image_180(image):
    return rotate_image_90(rotate_image_90(image))


def rotate_image_270(image):
    return rotate_image_90(rotate_image_90(rotate_image_90(image)))


def reverse_rotate_image_90(image):
    return rotate_image_270(image)


def reverse_rotate_image_180(image):
    return rotate_image_180(image)


def reverse_rotate_image_270(image):
    return rotate_image_90(image)


def flip_image(image):
    return np.fliplr(image.copy())


def reverse_flip_image(image):
    return flip_image(image)


def rotate(img, angle):

    if angle == 0:
        return img

    num_rows, num_cols = img.shape[:2]
    rotation_matrix = cv.getRotationMatrix2D((num_cols/2, num_rows/2), angle, 1)
    img_rotation = cv.warpAffine(img, rotation_matrix, (num_cols, num_rows))
    if(img_rotation.ndim == 2):
        img_rotation = np.expand_dims(img_rotation, axis = 2)

    return img_rotation


def get_output_directory_name(folder_name=params.folder_name):    
    return os.path.join('output-images', folder_name, str(params.scale)) + 'x'


def create_folders(folder_name): 
    directory_name = get_output_directory_name(folder_name)
    if not os.path.exists(directory_name):
       os.makedirs(directory_name)
       print('directory created: {} '.format(directory_name)) 
    else:
       print('directory {} exists '.format(directory_name))


def read_all_directory_images_from_directory_test(directory_path, add_to_path=None, read_black=True):
    '''
        This function reads the images from the directory_path (walk in every dir and read images).
        The output is list of nd-array (num_images, height, width, channels).
    '''
    if not os.path.exists(directory_path):
        print('Error!! Folder base name does not exit')
    folder_images = []
    folder_names = os.listdir(directory_path)
    for folder_name in folder_names:      
        images = []
        
        if add_to_path is None:
            images_path = os.path.join(directory_path, folder_name, '*' + params.image_ext) 
        else:
            images_path = os.path.join(directory_path, folder_name, add_to_path, '*' + params.image_ext) 
            
        files = glob.glob(images_path)
        num_images = len(files)
        print('There are {} images in {}'.format(num_images, images_path))
        # read the first image to get the size of the images
        image = cv.imread(files[0], cv.IMREAD_GRAYSCALE)
        print('The size of the first image is {}'.format(image.shape))
        
        images.append(np.expand_dims(image, 2))
        for index in range(1, num_images): 
            image = cv.imread(files[index], cv.IMREAD_GRAYSCALE)
            
            if read_black is False and image.max() == 0:
                continue
                
            images.append(np.expand_dims(image, 2))
            if(SHOW_IMAGES): 
                cv.imshow('image', image)
                cv.waitKey(0) 
        folder_images.append(np.array(images, 'uint8'))
        
    return folder_images
    
    
def read_all_directory_images_from_directory_test_depth(directory_path, add_to_path=None, list_idx=None):
    '''
        This function reads the images from the directory_path (walk in every dir and read images).
        The output is list of nd-array (num_images, height, width, channels).
    '''
    if not os.path.exists(directory_path):
        print('Error!! Folder base name does not exit')
    folder_images = []
    folder_names = os.listdir(directory_path)
    
    idx_to_read = [[] for i in range(len(folder_names))]
    j = -1
    for folder_name in folder_names:   
        j += 1
        images = []
        
        if add_to_path is None:
            images_path = os.path.join(directory_path, folder_name, '*' + params.image_ext) 
        else:
            images_path = os.path.join(directory_path, folder_name, add_to_path, '*' + params.image_ext) 
            
        files = glob.glob(images_path)
        num_images = len(files)
        print('There are {} images in {}'.format(num_images, images_path))
        # read the first image to get the size of the images
        image = cv.imread(files[0], cv.IMREAD_GRAYSCALE)
        print('The size of the first image is {}'.format(image.shape))
        
        images.append(np.expand_dims(image, 2))
        for index in range(1, num_images):  
            
            image = cv.imread(files[index], cv.IMREAD_GRAYSCALE)
            
            if image.max() == 0 and list_idx is None:
                continue
               
            if list_idx is None:
                idx_to_read[j].append(index)
            else:
                if index not in list_idx[j]:
                    continue
                    
            images.append(np.expand_dims(image, 2))
            if(SHOW_IMAGES): 
                cv.imshow('image', image)
                cv.waitKey(0) 
                
        folder_images.append(np.array(images, 'uint8'))
    
    if list_idx is None:
        return folder_images, idx_to_read
    else:
        return folder_images  


def read_all_directory_images_from_directory(directory_path):
    '''
        This function reads the images from the directory_path (walk in every dir and read images).
        The output is list with nd-array (num_images, height, width, channels) and the minimum btw the min height and min width.
    '''
    if not os.path.exists(directory_path):
        print('Error!! Folder base name does not exit')
        
    images = []
    min_H = 10000
    min_W = 10000
    folder_names = os.listdir(directory_path)
    for folder_name in folder_names:      
        images_path = os.path.join(directory_path, folder_name, '*' + params.image_ext) 
        files = glob.glob(images_path)
        num_images = len(files)
        print('There are {} images in {}'.format(num_images, images_path))
        # read the first image to get the size of the images
        image = cv.imread(files[0], cv.IMREAD_GRAYSCALE)
        
        if(image.shape[0] < min_H):
            min_H = image.shape[0]
        if(image.shape[1] < min_W):
            min_W = image.shape[1]
            
        print('The size of the first image is {}'.format(image.shape))
        
        images.append(np.expand_dims(image, 2))
        for index in range(1, num_images): 
            image = cv.imread(files[index], cv.IMREAD_GRAYSCALE)
            images.append(np.expand_dims(image, 2))
            if(SHOW_IMAGES): 
                cv.imshow('image', image)
                cv.waitKey(0) 
                
    return images, min(min_H, min_W)


def read_all_images_from_directory(images_path, return_np_array=True):
    '''
        This function reads the images from the directory specified in params.py.
        The output is a numpy ndarray of size (num_images, height, width, channels).
    ''' 
    if not os.path.exists(images_path):
        print('Error!! Folder  name does not exit')  
    if images_path[-1] != '/':
        images_path = images_path + '/'
        
    files = glob.glob(images_path + '*' + params.image_ext) 
    assert len(files) > 0
    num_images = len(files)
    print('There are {} images in {}'.format(num_images, images_path))
    # read the first image to get the size of the images
    image = cv.imread(files[0], cv.IMREAD_GRAYSCALE)
    print('The size of the first image is {}'.format(image.shape))
    images = []
    
    images.append(np.expand_dims(image, 2))
    for index in range(1, num_images): 
        image = cv.imread(files[index], cv.IMREAD_GRAYSCALE)
        images.append(np.expand_dims(image, 2))
        if(SHOW_IMAGES): 
            cv.imshow('image', image)
            cv.waitKey(0)
        
    if(not return_np_array):
        return images
        
    return np.array(images) 


def read_all_patches_from_directory(base_dir, folder='', return_np_array=True):
    '''
        This function reads the images from the base_dir (walk in every dir named folder and read images).
        The output is list with nd-array (num_images, height, width, channels) and the minimum btw the min height and min width.
    '''
    if not os.path.exists(base_dir):
        print('Error!! Folder base name does not exit')
        
    images = [] 
    folder_names = os.listdir(base_dir)  
    for folder_name in folder_names:      
        
        images_path = os.path.join(base_dir, folder_name, folder, '*' + params.image_ext)  
        files = glob.glob(images_path)
        num_images = len(files)
        print('There are {} images in {}'.format(num_images, images_path))
        # read the first image to get the size of the images
        image = cv.imread(files[0], cv.IMREAD_GRAYSCALE)
 
            
        print('The size of the first image is {}'.format(image.shape))
        
        images.append(np.expand_dims(image, 2))
        for index in range(1, num_images): 
            image = cv.imread(files[index], cv.IMREAD_GRAYSCALE)
            images.append(np.expand_dims(image, 2))
            if(SHOW_IMAGES): 
                cv.imshow('image', image)
                cv.waitKey(0) 
                
    if(not return_np_array):
        return images
        
    return np.array(images) 


def resize_height_width_image_standard(image, factor, interpolation_method = cv.INTER_LINEAR):
    new_width = image.shape[1] * factor
    new_heigth = image.shape[0] * factor
    return cv.resize(image, (new_width, new_heigth), interpolation = interpolation_method)


def resize_height_width_3d_image_standard(images, new_heigth, new_width, interpolation_method = cv.INTER_LINEAR):  
    '''
        images is a nd-array of size (num_images, new_heigth, new_width, num_channels) and this function resize every image from the input.
    '''
    num_images = images.shape[0] 
    num_channels = images.shape[-1]
    resized_images = np.zeros((num_images, new_heigth, new_width, num_channels))
 
    for index in range(num_images):
        image = images[index, :, :, :]
        if(num_channels == 1):
            resized_images[index, :, :, 0] = cv.resize(image, (new_width, new_heigth), interpolation = interpolation_method)
        else:
            resized_images[index, :, :, :] = cv.resize(image, (new_width, new_heigth), interpolation = interpolation_method)
            
        if(SHOW_IMAGES):
            cv.imshow('image',  resized_images[index, :, :, :] / 255)
            cv.waitKey(0)  
    
    return resized_images


def resize_height_width_list_of_3d_image_standard(image_list, scale, interpolation_method = cv.INTER_LINEAR):
    '''
        Resize a list of 3d images on each dimensions: depth, height, width. 
    ''' 
    new_list = []
    for item in image_list: 
        d, h, w = item.shape[:3] 
        new_list.append(resize_height_width_3d_image_standard(item, int(h * scale), int(w * scale), interpolation_method)) 
        
    return new_list   


def resize_list_of_3d_image_standard(image_list, scale, interpolation_method = cv.INTER_LINEAR):
    '''
        Resize a list of 3d images on each dimensions: depth, height, width. 
    '''
    new_list = []
    for item in image_list: 
        d, h, w = item.shape[:3] 
        new_list.append(resize_3d_image_standard(item, int(d * scale),  int(h * scale), int(w * scale), interpolation_method))
        
    return new_list
    

def resize_depth_3d_image_standard(images, new_depth, height, width, interpolation_method = cv.INTER_LINEAR):
    '''
        Resize the depth of the 3d image by taking every coord as a depth row and resizing it.
    '''
    resized_3d_images = np.zeros((new_depth, height, width, images.shape[3]))       
    for y in range(height):
        for x in range(width):
            depth_row = images[:, y, x]
            resized_depth_row = cv.resize(depth_row, (1, new_depth), interpolation = interpolation_method)
            resized_3d_images[:, y, x, 0] = resized_depth_row.ravel()
            
    if(SHOW_IMAGES):    
        for index in range(new_depth):
            image = resized_3d_images[index, :, :, :] 
            cv.imshow('image', image / 255)
            cv.waitKey(0)            
     
    return resized_3d_images


def resize_3d_image_standard(images, new_depth, new_height, new_width, interpolation_method = cv.INTER_LINEAR): 
     
    resized_images = resize_height_width_3d_image_standard(images, new_height, new_width, interpolation_method)
    resized_3d_images = resize_depth_3d_image_standard(resized_images, new_depth, new_height, new_width, interpolation_method)
       
    return resized_3d_images


def write_3d_images(path_images, images, prefix):
    '''
        This function writes the images in the directory specified in params.py with the prefix specified as a param.
        The input is a numpy ndarray of size (num_images, height, width, channels) and a string.
    ''' 
    num_images = images.shape[0]
    folder_name = path_images.split('/')[-2]
    directory_name = os.path.join(get_output_directory_name(folder_name), prefix)
    if not os.path.exists(directory_name):
       os.makedirs(directory_name)
       print('directory created: {} '.format(directory_name)) 
    for index in range(num_images):
        image = images[index, :, :, :] 
        cv.imwrite(os.path.join(directory_name, str(index) + '.' + params.image_ext), image)
    

def flip_images(images):
    num_images = images.shape[0]   
    flipped_images = np.zeros(images.shape)
    for index in range(num_images):
        image = images[index, :, :, 0]  
        flipped_images[index, :, :, 0] = cv.flip(image, 1)
        
    return flipped_images


def rotate_images(images, angle):
    num_images = images.shape[0]   
    rotated_images = np.zeros(images.shape)
    for index in range(num_images):
        image = images[index, :, :, 0]  
        rotated_images[index, :, :, 0] = rotate(image, angle)
        
    return rotated_images
    