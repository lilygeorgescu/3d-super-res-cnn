#  Copyright (c) 2020. Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Nicolae Verga
#  Convolutional Neural Networks with  Intermediate Loss for 3D Super-Resolution of CT and MRI Scans
#  Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
#  (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#

from utils import *
import params
import tensorflow as tf
import pdb
import cv2 as cv 


def trim_image(image):
    image[image > 255] = 255
    return image


def run_network(downscaled_image, checkpoint):
 
    # network for original image
    config = tf.ConfigProto(
            device_count={'GPU': 1}
        ) 
    
    graph_or = tf.Graph()
    sess_or = tf.Session(graph=graph_or, config=config)
    with graph_or.as_default():
        input_or = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params.num_channels), name='input_or')  
        _, output_or = params.network_architecture(input_or)
        saver = tf.train.Saver()
        saver.restore(sess_or, checkpoint)

    graph_tr = tf.Graph()
    sess_tr = tf.Session(graph=graph_tr, config=config)
    with graph_tr.as_default():
        input_tr = tf.placeholder(tf.float32, (1, downscaled_image.shape[2], downscaled_image.shape[1], params.num_channels), name='input_tr')  
        _, output_tr = params.network_architecture(input_tr, reuse=False)
        saver = tf.train.Saver()
        saver.restore(sess_tr, checkpoint)
    
    num_images = downscaled_image.shape[0]
    cnn_output = []
    for image in downscaled_image:
        out_images = [] 
        
        # original 0 
        res = trim_image(sess_or.run(output_or, {input_or: [image]})[0])
        out_images.append(res)
        
        # flip 0 
        res = trim_image(sess_or.run(output_or, {input_or: [flip_image(image)]})[0])
        out_images.append(reverse_flip_image(res))
         
        # original 180
        rot180_image = rotate_image_180(image)
        res = trim_image(sess_or.run(output_or, {input_or: [rot180_image]})[0])
        out_images.append(reverse_rotate_image_180(res)) 
        
        # flip 180 
        res = trim_image(sess_or.run(output_or, {input_or: [flip_image(rot180_image)]})[0])
        out_images.append(reverse_rotate_image_180(reverse_flip_image(res)))          
       
        if use_mean:
            cnn_output.append(np.round(np.mean(np.array(out_images), axis=0)))
        else:
            cnn_output.append(np.round(np.median(np.array(out_images), axis=0)))
        
    cnn_output = np.array(cnn_output) 
 
    return cnn_output


def read_images(test_path):

    if transposed_2_1:
        add_to_path_gt = 'transposed_2_1'
        add_to_path_in = 'input_d_2_1_x%d' % scale 
    else:
        add_to_path_gt = 'transposed'
        add_to_path_in = 'input_d_x%d' % scale
        
    test_images, lists_idx = read_all_directory_images_from_directory_test_depth(test_path, add_to_path=add_to_path_in)  
    test_images_gt = read_all_directory_images_from_directory_test_depth(test_path, add_to_path=add_to_path_gt, list_idx=lists_idx)
    print(len(test_images), len(test_images_gt))    

    return test_images_gt, test_images 


def predict(downscaled_image, original_image, checkpoint): 

    scale_factor = params.scale    
    # standard_resize = resize_height_width_3d_image_standard(downscaled_image, int(downscaled_image.shape[1]), int(downscaled_image.shape[2])*scale_factor, interpolation_method = params.interpolation_method)  
    cnn_output = run_network(downscaled_image, checkpoint)  
    print(cnn_output.shape, original_image.shape)
    # print(standard_resize.shape, original_image.shape)
    ssim_cnn, psnr_cnn = compute_ssim_psnr_batch(cnn_output, original_image)
    # ssim_standard, psnr_standard = compute_ssim_psnr_batch(standard_resize, original_image)

    return ssim_cnn, psnr_cnn, 0, 0 
    
    
def compute_performance_indices(test_images_gt, test_images, checkpoint):

    num_images = 0 
    ssim_cnn_sum = 0; psnr_cnn_sum = 0; ssim_standard_sum = 0; psnr_standard_sum = 0;  
 
    for index in range(len(test_images_gt)):  
            
        ssim_cnn, psnr_cnn, ssim_standard, psnr_standard = predict(test_images[index], test_images_gt[index], checkpoint)
        tf.reset_default_graph()
        ssim_cnn_sum += ssim_cnn; psnr_cnn_sum += psnr_cnn 
        ssim_standard_sum += ssim_standard; psnr_standard_sum += psnr_standard 
        num_images += test_images_gt[index].shape[0]
     
    print('standard {} --- psnr = {} ssim = {}'.format(test_path, psnr_standard_sum/num_images, ssim_standard_sum/num_images)) 
    print('cnn {} --- psnr = {} ssim = {}'.format(test_path, psnr_cnn_sum/num_images, ssim_cnn_sum/num_images))
    

 
    
# checkpoint = tf.train.latest_checkpoint(params.folder_data)    
checkpoint = './data_ckpt/model.ckpt8'
use_mean = False 

scale = 2 
      
test_path = 'C:\\Research\\SR\\medical images\\namic\\images-testing\\t1w' 
train_path = './data/train' 

path_used = test_path

transposed_2_1 = False
test_images_gt_2_1, test_images_2_1 = read_images(path_used)

transposed_2_1 = True
test_images_gt, test_images = read_images(path_used)

test_images_gt += test_images_gt_2_1
test_images += test_images_2_1
 
compute_performance_indices(test_images_gt, test_images, checkpoint)