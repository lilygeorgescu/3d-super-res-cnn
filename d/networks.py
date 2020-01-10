#  Copyright (c) 2020. Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Nicolae Verga
#  Convolutional Neural Networks with  Intermediate Loss for 3D Super-Resolution of CT and MRI Scans
#  Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
#  (https://creativecommons.org/licenses/by-nc-sa/4.0/)
#

import tensorflow as tf
import params
import numpy as np
import pdb

    
def _phase_shift_D(I, r):
    # Helper function with main phase shift operation 
    bsize, a, b, c = I.get_shape().as_list()
    X = tf.reshape(I, (bsize, a, b, 1, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1    
    if X.shape[0] == 1:
            X = tf.split(X, a, 1)  # a, [bsize, b, r, r] 
            X = tf.concat([tf.expand_dims(tf.squeeze(x), 0) for x in X], 1)  # bsize, b, a*r, r 
            X = tf.reshape(X, [X.shape[0].value, X.shape[1].value, X.shape[2].value, 1])            
            X = tf.split(X, b, 1)  # b, [bsize, a*r, r] 
            X = tf.concat([tf.expand_dims(tf.squeeze(x), 0) for x in X], 1)
    else: 
            X = tf.split(X, a, 1)  # a, [bsize, b, r, r] 
            X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, b, a*r, r 
            X = tf.reshape(X, [X.shape[0].value, X.shape[1].value, X.shape[2].value, 1])            
            X = tf.split(X, b, 1)  # b, [bsize, a*r, r] 
            X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, a*r, b*r 
             
        
    return tf.reshape(X, (bsize, a*1, b*r, 1))    
    
def PS_D(X, r): 
    X = _phase_shift_D(X, r)
    return X  


def SRCNN_late_upscaling_D(im, reuse=False, is_training=False): 

 	with tf.name_scope('depth_net') as scope:  
    
            output_1 = tf.layers.conv2d(im, filters=32, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu, reuse=reuse, name='depth_net/conv2d')
         
            
        # residual block
            output_2 = tf.layers.conv2d(output_1, filters=32, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu, reuse=reuse, name='depth_net/conv2d_1') 
            output_3 = tf.layers.conv2d(output_2, filters=32, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu, reuse=reuse, name='depth_net/conv2d_2')
            output_4 = tf.add(tf.multiply(output_1 , 1), output_3)
            output = tf.layers.conv2d(output_4, filters=32, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu, reuse=reuse, name='depth_net/conv2d_3')
            output = tf.layers.conv2d(output, filters=32, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu, reuse=reuse, name='depth_net/conv2d_4')
            output = tf.add(tf.multiply(output_1, 1), output)
            feature_map_for_ps = tf.layers.conv2d(output, filters=params.num_channels * params.scale, kernel_size=(3, 3), strides=1, padding='SAME', activation=tf.nn.relu, reuse=reuse, name='depth_net/conv2d_5')  

            output_PS = PS_D(feature_map_for_ps, params.scale)  
        
            output_5 = tf.layers.conv2d(output_PS, filters=32, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu, name='depth_net/last_layer_1', reuse=reuse)
              
            output_6 = tf.layers.conv2d(output_5, filters=32, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu, name='depth_net/last_layer_2', reuse=reuse)
            output_7 = tf.layers.conv2d(output_6, filters=32, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu, name='depth_net/last_layer_3', reuse=reuse)
            output_8 = tf.add(tf.multiply(output_5, 1), output_7, name='depth_net/last_layer_4')

            output_9 = tf.layers.conv2d(output_8, filters=params.num_channels, kernel_size=3, strides=1, padding='SAME', activation=tf.nn.relu, name='depth_net/last_layer_5', reuse=reuse)
      
 	return output_PS, output_9
