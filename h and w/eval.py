#  Copyright (c) 2020. Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Nicolae Verga
#  Convolutional Neural Networks with  Intermediate Loss for 3D Super-Resolution of CT and MRI Scans
#  Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
#  (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#

import tensorflow as tf
import numpy as np
import cv2 as cv 
import params
import utils
import pdb
import re
import os 

params.show_params()

config = tf.ConfigProto(
        device_count={'GPU': 1}
    ) 


def upscale(downscaled_image, checkpoint):

    scale_factor = params.scale   
     
    # cnn resize  
    input = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params.num_channels), name='input')  
    _, output = params.network_architecture(input) 

    with tf.Session(config=config) as sess:  
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print('restoring from ' + checkpoint)
        saver.restore(sess, checkpoint)
         
        # step 1 - apply cnn on each resized image, maybe as a batch 
        cnn_output = []
        for image in downscaled_image: 
            cnn_output.append(sess.run(output, feed_dict={input: [image]})[0])
    
        cnn_output = np.array(cnn_output)   
        cnn_output = np.round(cnn_output) 
        cnn_output[cnn_output > 255] = 255 
        
        return cnn_output
        
        
def predict(downscaled_image, original_image, checkpoint):
 
    downscaled_image = upscale(downscaled_image, checkpoint)
    tf.reset_default_graph()
         
    ssim_cnn, psnr_cnn = utils.compute_ssim_psnr_batch(downscaled_image, original_image) 

    return ssim_cnn, psnr_cnn
    
        
def read_images(test_path):

    test_images_gt = utils.read_all_directory_images_from_directory_test(test_path, add_to_path='original')
    test_images = utils.read_all_directory_images_from_directory_test(test_path, add_to_path='input_x%d' % scale) 
    return test_images_gt, test_images


def compute_performance_indices(test_path, test_images_gt, test_images, checkpoint, write_to_summary=True):

    num_images = 0 
    ssim_cnn_sum = 0; psnr_cnn_sum = 0; ssim_standard_sum = 0; psnr_standard_sum = 0;  
 
    for index in range(len(test_images)):
        # pdb.set_trace()
        ssim_cnn, psnr_cnn = predict(test_images[index], test_images_gt[index], checkpoint)
        tf.reset_default_graph()
        ssim_cnn_sum += ssim_cnn; psnr_cnn_sum += psnr_cnn  
        num_images += test_images[index].shape[0]
      
    print('cnn {} --- psnr = {} ssim = {}'.format(test_path, psnr_cnn_sum/num_images, ssim_cnn_sum/num_images))
    
    if test_path.find('test') != -1 and write_to_summary is True:
     
        tf.summary.scalar('psnr_cnn', psnr_cnn_sum/num_images)   
        tf.summary.scalar('ssim_cnn', ssim_cnn_sum/num_images)  
        merged = tf.summary.merge_all() 
        writer = tf.summary.FileWriter('test.log')  
        epoch = re.findall(r'\d+', checkpoint)
        epoch = int(epoch[0]) 
        with tf.Session(config=config) as sess:
            merged_ = sess.run(merged)
            writer.add_summary(merged_, epoch)


test_path = './../../../../images-testing/t1w' 
eval_path = './data/train'
scale = 2

test_images_gt, test_images = read_images(test_path)  
checkpoint = tf.train.latest_checkpoint(params.folder_data)
compute_performance_indices(test_path, test_images_gt, test_images, checkpoint, write_to_summary=False)

 