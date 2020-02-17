#  Copyright (c) 2020. Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Nicolae Verga
#  Convolutional Neural Networks with  Intermediate Loss for 3D Super-Resolution of CT and MRI Scans
#  Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
#  (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#

import cv2 as cv
import numpy as np
import tensorflow as tf

from utils import *
import utils
import params
import pdb

config = tf.ConfigProto(
        device_count={'GPU': 1}
    ) 


def trim_image(image):
    image[image > 255] = 255
    return image
    
    
def upscale_h_w(downscaled_image, checkpoint):
 
    # network for original image
    config = tf.ConfigProto(
            device_count={'GPU': 1}
        ) 
    
    graph_or = tf.Graph()
    sess_or = tf.Session(graph=graph_or, config=config)
    with graph_or.as_default():
        input_or = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params.num_channels), name='input_or')
        if is_v2:
            _, output_or = nets_hs.SRCNN_late_upscaling_H_W(input_or) 
        else:
            output_or = nets_hs.SRCNN_late_upscaling_H_W(input_or)
        saver = tf.train.Saver()
        saver.restore(sess_or, checkpoint)

    graph_tr = tf.Graph()
    sess_tr = tf.Session(graph=graph_tr, config=config)
    with graph_tr.as_default():
        input_tr = tf.placeholder(tf.float32, (1, downscaled_image.shape[2], downscaled_image.shape[1], params.num_channels), name='input_tr')  
        if is_v2:
            _, output_tr = nets_hs.SRCNN_late_upscaling_H_W(input_tr, reuse=False)
        else:
            output_tr = nets_hs.SRCNN_late_upscaling_H_W(input_tr, reuse=False)
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
        
        # original 90
        rot90_image = rotate_image_90(image)
        res = trim_image(sess_tr.run(output_tr, {input_tr: [rot90_image]})[0])
        out_images.append(reverse_rotate_image_90(res)) 
        
        # flip 90 
        res = trim_image(sess_tr.run(output_tr, {input_tr: [flip_image(rot90_image)]})[0])
        out_images.append(reverse_rotate_image_90(reverse_flip_image(res)))   

        # original 180
        rot180_image = rotate_image_180(image)
        res = trim_image(sess_or.run(output_or, {input_or: [rot180_image]})[0])
        out_images.append(reverse_rotate_image_180(res)) 
        
        # flip 180 
        res = trim_image(sess_or.run(output_or, {input_or: [flip_image(rot180_image)]})[0])
        out_images.append(reverse_rotate_image_180(reverse_flip_image(res)))          
        
        # original 270
        rot270_image = rotate_image_270(image)
        res = trim_image(sess_tr.run(output_tr, {input_tr: [rot270_image]})[0])
        out_images.append(reverse_rotate_image_270(res)) 
        
        # flip 270 
        res = trim_image(sess_tr.run(output_tr, {input_tr: [flip_image(rot270_image)]})[0])
        out_images.append(reverse_rotate_image_270(reverse_flip_image(res))) 
        # pdb.set_trace()
        if use_mean:
            cnn_output.append(np.round(np.mean(np.array(out_images), axis=0)))
        else:
            cnn_output.append(np.round(np.median(np.array(out_images), axis=0)))
        
    cnn_output = np.array(cnn_output)
     
    return cnn_output    


def resize_h_w(downscaled_image, original_image=None):
    cnn_output = upscale_h_w(downscaled_image, checkpoint_h_w)
    if original_image is not None:
        ssim_cnn, psnr_cnn = utils.compute_ssim_psnr_batch(cnn_output, original_image)
        return ssim_cnn, psnr_cnn
    else:
        return cnn_output


def predict_1_2(downscaled_image, checkpoint):
    # network for original image
    config = tf.ConfigProto(
            device_count={'GPU': 1}
        ) 
    
    graph_or = tf.Graph()
    sess_or = tf.Session(graph=graph_or, config=config)
    with graph_or.as_default():
        input_or = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params.num_channels), name='input_or')  
        _, output_or = nets_d.SRCNN_late_upscaling_D(input_or)
        saver = tf.train.Saver()
        saver.restore(sess_or, checkpoint)

    graph_tr = tf.Graph()
    sess_tr = tf.Session(graph=graph_tr, config=config)
    with graph_tr.as_default():
        input_tr = tf.placeholder(tf.float32, (1, downscaled_image.shape[2], downscaled_image.shape[1], params.num_channels), name='input_tr')  
        _, output_tr = nets_d.SRCNN_late_upscaling_D(input_tr, reuse=False)
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
    print('one image done')
    cnn_output = np.array(cnn_output)
    cnn_output = np.transpose(cnn_output, [2, 0, 1, 3])   
 
    return cnn_output


def predict_2_1(downscaled_image, checkpoint):
 
    # network for original image
    config = tf.ConfigProto(
            device_count={'GPU': 1}
        ) 
    
    graph_or = tf.Graph()
    sess_or = tf.Session(graph=graph_or, config=config)
    with graph_or.as_default():
        input_or = tf.placeholder(tf.float32, (1, downscaled_image.shape[1], downscaled_image.shape[2], params.num_channels), name='input_or')  
        _, output_or = nets_d.SRCNN_late_upscaling_D(input_or)
        saver = tf.train.Saver()
        saver.restore(sess_or, checkpoint)

    graph_tr = tf.Graph()
    sess_tr = tf.Session(graph=graph_tr, config=config)
    with graph_tr.as_default():
        input_tr = tf.placeholder(tf.float32, (1, downscaled_image.shape[2], downscaled_image.shape[1], params.num_channels), name='input_tr')  
        _, output_tr = nets_d.SRCNN_late_upscaling_D(input_tr, reuse=False)
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
    cnn_output = np.transpose(cnn_output, [2, 1, 0, 3])   
 
    return cnn_output


def predict_d(downscaled_image_1_2, downscaled_image_2_1, checkpoint): 

    scale_factor = params.scale
    image_1_2 = predict_1_2(downscaled_image_1_2, checkpoint)
    tf.reset_default_graph()
    image_2_1 = predict_2_1(downscaled_image_2_1, checkpoint)  
    cnn_output = 0.5 * (image_1_2 + image_2_1)
           
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
    cnn_output = predict_d(downscaled_image.transpose([1, 2, 0, 3]), downscaled_image.transpose([2, 1, 0, 3]), checkpoint_d)  
    # utils.write_3d_images(image_names.pop(0), cnn_output, 'cnn')    
    if original_image is not None:
        ssim_cnn, psnr_cnn = utils.compute_ssim_psnr_batch(cnn_output, original_image)
        return ssim_cnn, psnr_cnn
    else:
        return cnn_output


def compute_performance_indices(test_images_gt, test_images):

    num_images = 0 
    ssim_cnn_sum = 0; psnr_cnn_sum = 0; 
 
    for index in range(len(test_images)):
        # pdb.set_trace()
        if use_hw_d:
            ssim_cnn, psnr_cnn = resize_depth(resize_h_w(test_images[index]), test_images_gt[index])
        else: 
            
            ssim_cnn, psnr_cnn = resize_h_w(resize_depth(test_images[index]), test_images_gt[index])
            
        tf.reset_default_graph()
        ssim_cnn_sum += ssim_cnn; psnr_cnn_sum += psnr_cnn  
        num_images += test_images_gt[index].shape[0]
      
        print('cnn {} --- psnr = {} ssim = {}'.format(test_path, psnr_cnn_sum/num_images, ssim_cnn_sum/num_images))


def read_images(test_path): 

    if use_hw_d:
        add_to_path = 'input_hw_d_%d' % scale 
    else:
        add_to_path = 'input'
        
    test_images_gt = utils.read_all_directory_images_from_directory_test(test_path, add_to_path='original')
    test_images = utils.read_all_directory_images_from_directory_test(test_path, add_to_path=add_to_path)
    
    return test_images_gt, test_images
    
use_hw_d = True
test_path = 'C:\\Research\\SR\\medical images\\namic\\images-testing\\t1w'  


use_standard_d = False


is_v2 = True

# from D_v1 import networks as nets_d
# from HW import networks as nets_hs
# checkpoint_d = './D_v1/data_ckpt/model.ckpt1'
# checkpoint_h_w = './HW/data_ckpt/model.ckpt128'
# cnn ./data/test --- psnr = 32.21869619109745 ssim = 0.8868613551347508
# cnn ./data/test --- psnr = 32.21240828089162 ssim = 0.8865181006890493



# from D_v2_32 import networks as nets_d
# from HW_v2 import networks as nets_hs
# checkpoint_d = './D_v2_32/data_ckpt/model.ckpt2'
# checkpoint_h_w = './HW_v2/data_ckpt/model.ckpt35'
# cnn ./data/test --- psnr = 32.234210332680746 ssim = 0.8861769604627088
# cnn ./data/test --- psnr = 32.248339714641226 ssim = 0.8869433580870602


 

use_mean = False 
from D_v2 import networks as nets_d
from HW_v2 import networks as nets_hs
checkpoint_d = './D_v2/data_ckpt/model.ckpt8'
checkpoint_h_w = './HW_v2/data_ckpt/model.ckpt37' 
scale = 2
# cnn ./data/test --- psnr = 32.09764710790664 ssim = 0.888204505949677
# cnn ./data/test --- psnr = 32.08028303636246 ssim = 0.8874917792189498


# use_mean = True
# scale = 4
# from D_v2_4 import networks as nets_d
# from HW_v2_4 import networks as nets_hs
# checkpoint_d = './D_v2_4/data_ckpt/model.ckpt3'
# checkpoint_h_w = './HW_v2_4/data_ckpt/model.ckpt43'

# cnn ./data/test --- psnr = 29.539805633557354 ssim = 0.7868711608337497
# cnn ./data/test --- psnr = 29.539128557872 ssim = 0.7871700061539182
test_images_gt, test_images = read_images(test_path) 


image_names = ['00001_0007', '00001_0009', '00001_0010', '00001_0011']
compute_performance_indices(test_images_gt, test_images)










