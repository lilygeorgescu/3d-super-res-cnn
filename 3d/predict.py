#  Copyright (c) 2020. Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Nicolae Verga
#  Convolutional Neural Networks with  Intermediate Loss for 3D Super-Resolution of CT and MRI Scans
#  Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
#  (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#

import cv2 as cv
import numpy as np
import tensorflow as tf
 
import utils
import params
import pdb

config = tf.ConfigProto(
        device_count = {'GPU': 1}
    ) 
 
def resize_h_w(downscaled_image, original_image=None):    
    tf.reset_default_graph()         
    # cnn resize  
    input = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params.num_channels), name='input')  
    
    if is_v2:
        _, output = nets_hs.SRCNN_late_upscaling_H_W(input) 
    else:
        output = nets_hs.SRCNN_late_upscaling_H_W(input) 

    with tf.Session(config=config) as sess:  
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print('restoring from ' + checkpoint_h_w)
        saver.restore(sess, checkpoint_h_w)
         
        # step 1 - apply cnn on each resized image, maybe as a batch 
        cnn_output = []
        for image in downscaled_image: 
            cnn_output.append(sess.run(output, feed_dict={input: [image]})[0])
    
        cnn_output = np.array(cnn_output)   
        cnn_output = np.round(cnn_output) 
        cnn_output[cnn_output > 255] = 255
        
        if original_image is not None:
            ssim_cnn, psnr_cnn = utils.compute_ssim_psnr_batch(cnn_output, original_image)
            return ssim_cnn, psnr_cnn
        else:
            return cnn_output
        
def resize_depth(downscaled_image, original_image=None):

    if use_standard_d:
        out = utils.resize_depth_3d_image_standard(downscaled_image, (downscaled_image.shape[0] * params.scale), downscaled_image.shape[1], downscaled_image.shape[2], cv.INTER_LINEAR) 
        if original_image is not None:
            ssim_cnn, psnr_cnn = utils.compute_ssim_psnr_batch(out, original_image)
            return ssim_cnn, psnr_cnn
        else:
            return out
    
    tf.reset_default_graph()
    scale_factor = params.scale    
    
    downscaled_image = image_3d = np.transpose(downscaled_image, [1, 2, 0, 3])
    # cnn resize  
    input = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params.num_channels), name='input')  
    
 
    _, output = nets_d.SRCNN_late_upscaling_D(input)
   
        
    with tf.Session(config=config) as sess:  
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print('restoring from ' + checkpoint_d)
        saver.restore(sess, checkpoint_d)
         
        # step 1 - apply cnn on each resized image, maybe as a batch 
        cnn_output = []
        for image in downscaled_image: 
            cnn_output.append(sess.run(output, feed_dict={input: [image]})[0])
    
        cnn_output = np.array(cnn_output)   
        cnn_output = np.round(cnn_output) 
        cnn_output[cnn_output > 255] = 255 
        cnn_output = np.transpose(cnn_output, [2, 0, 1, 3])  
            
        if original_image is not None:
            ssim_cnn, psnr_cnn = utils.compute_ssim_psnr_batch(cnn_output, original_image)
            return ssim_cnn, psnr_cnn
        else:
            return cnn_output


def compute_performance_indeces(test_images_gt, test_images):

    num_images = 0 
    ssim_cnn_sum = 0; psnr_cnn_sum = 0; 
 
    for index in range(len(test_images)):
        # pdb.set_trace()
        # if use_hw_d:
        ssim_cnn, psnr_cnn = resize_depth(resize_h_w(test_images[index]), test_images_gt[index])
        # else: 
            
            # ssim_cnn, psnr_cnn = resize_h_w(resize_depth(test_images[index]), test_images_gt[index])
            
        tf.reset_default_graph()
        ssim_cnn_sum += ssim_cnn; psnr_cnn_sum += psnr_cnn  
        num_images += test_images_gt[index].shape[0]
      
        print('cnn {} --- psnr = {} ssim = {}'.format(test_path, psnr_cnn_sum/num_images, ssim_cnn_sum/num_images))
  
def read_images(test_path): 
 
    add_to_path = 'input_hw_d_%d' % scale 
        
    test_images_gt = utils.read_all_directory_images_from_directory_test(test_path, add_to_path='original')
    test_images = utils.read_all_directory_images_from_directory_test(test_path, add_to_path=add_to_path)
    
    return test_images_gt, test_images
     
test_path = 'C:\\Research\\SR\\medical images\\namic\\images-testing\\t1w'  


use_standard_d = False 

assert use_standard_d is False

scale = 2
is_v2 = True


from D_v2 import networks as nets_d
from HW_v2 import networks as nets_hs
checkpoint_d = './D_v2/data_ckpt/model.ckpt8'
checkpoint_h_w = './HW_v2/data_ckpt/model.ckpt37' 

test_images_gt, test_images = read_images(test_path)  
compute_performance_indeces(test_images_gt, test_images)



 









