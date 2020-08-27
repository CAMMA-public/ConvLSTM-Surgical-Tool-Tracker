#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weakly-supervised ConvLSTM Surgical Tool Tracker
================
------
**A re-implementation of surgical tool tracker in**<br>
<i>Nwoye, C. I., Mutter, D., Marescaux, J., & Padoy, N. (2019). 
    Weakly supervised convolutional LSTM approach for tool tracking in laparoscopic videos. 
    International journal of computer assisted radiology and surgery, 14(6), 1059-1067.<br></i>
    <b> Cite this paper, if you use this code or part of it.. </b>
    
    
(c) Research Group CAMMA, University of Strasbourg, France<br>
Website: http://camma.u-strasbg.fr<br>
Code author: Chinedu Nwoye <br>
    
-----
"""


import tensorflow as tf
import numpy as np
import lib.resnet_utils as utils

    
#%% 1. Detection Model for tools presence
        
class ResNet(object):
    def __init__(self, images, version=18, is_training=True, num_gpus=1):    
        self._images                = images 
        self._is_training           = is_training
        self._version               = version
        self._num_gpus              = num_gpus   
        self._counted_scope         = []
        self._flops                 = 0
        self._weights               = 0     
        
        '''Initialize the parameters for the model.
        Args:
            images:         4D input tensor [batch_size, height, width, channel] of image of type int32, int64, float32 or float64.
            batch_size:     Interger, number of examples in a batch, also use as timesteps in LSTM models
            is_train:       Boolean, (optional), set True for training and False for validation and testing. Default: True
            num_gpus:       Integer, not used. Default: 1
        '''        
        
        
    def _networks_map(self, version=161):
        maps = {
            18  : {'blocks': [2, 2, 2 , 2],},
            34  : {'blocks': [3, 4, 6 , 3],},
            50  : {'blocks': [3, 4, 6 , 3],},
            101 : {'blocks': [3, 4, 23, 3],},
            152 : {'blocks': [3, 8, 36, 3],},
        }
        return maps.get(version)

        
    #%% Build model in graph   
    def _build_model(self):
        ''' Build the deep learning network for tools detection and/or tracking based on the selected model
        Args: super(Model)
        Returns: 
            f_map7:     Float, [batch_size, height, width, num_classes] channel localization map, one channel corresponding to each tool class in the order of training.
        '''
                  
        resnet_version = self._networks_map(self._version)     
        blocks         = resnet_version.get('blocks')
        trees          = dict()
        print('Model blocks: ', blocks)
    
        filters = [64, 64, 128, 256, 512]
        kernels = [7,   3,   3,   3,   3]
        strides = [2,   0,   2,   1,   1]
        print('\tReceiving image:: {}'.format( self._images.get_shape() ))

        with tf.name_scope('ResNet'): 
            print('Constructing ResNet backbone:')
            trees['block_0'] = self._images 
            # conv1
            with tf.variable_scope('conv1'):                 
                x = self._conv(self._images, kernels[0], filters[0], strides[0])
                x = self._bn(x)
                x = self._relu(x)
                x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')  
                trees['block_1'] = x   
                print('\tBuilding units: conv1 -> {}'.format( x.get_shape() ))  
                
            # conv2_x
            x = self._residual_block(x, name='conv2_1')
            x = self._residual_block(x, name='conv2_2')
            trees['block_2'] = x   
    
            # conv3_x
            x = self._residual_block_first(x, filters[2], strides[2], name='conv3_1')
            x = self._residual_block(x, name='conv3_2')
            trees['block_3'] = x   
    
            # conv4_x
            x = self._residual_block_first(x, filters[3], strides[3], name='conv4_1')
            x = self._residual_block(x, name='conv4_2')
            trees['block_4'] = x   
    
            # conv5_x
            x = self._residual_block_first(x, filters[4], strides[4], name='conv5_1')
            x = self._residual_block(x, name='conv5_2')  
            trees['block_5'] = x   
        return x

    
                
    #%% Helper functions
    
    def _residual_block_first(self, x, out_channel, strides, name="unit"):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name,reuse=None) as scope:
            print('\tBuilding unit: {}: {}'.format( scope.name, x.get_shape() ))
            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
            else:
                shortcut = self._conv(x, 1, out_channel, strides, name='shortcut')
            # Residual
            x = self._conv(x, 3, out_channel, strides, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, out_channel, 1, name='conv_2')
            x = self._bn(x, name='bn_2')
            # Merge
            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x    


    def _residual_block(self, x, input_q=None, output_q=None, name="unit"):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name,reuse=None) as scope:
            print('\tBuilding unit: {}: {}'.format( scope.name, x.get_shape() ))
            # Shortcut connection
            shortcut = x
            # Residual
            x = self._conv(x, 3, num_channel, 1, input_q=input_q, output_q=output_q, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, num_channel, 1, input_q=output_q, output_q=output_q, name='conv_2')
            x = self._bn(x, name='bn_2')
            # Merge
            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x    
    
    
    def _bn(self, x, name="bn"):
        x = utils._bn(x, self._is_training, None, name)
        return x
    

    def _relu(self, x, name="relu"):
        x = utils._relu(x, 0.0, name)
        return x
    

    def _get_data_size(self, x):
        return np.prod(x.get_shape().as_list()[1:])
    

    def _add_flops_weights(self, scope_name, f, w):
        if scope_name not in self._counted_scope:
            self._flops += f
            self._weights += w
            self._counted_scope.append(scope_name)
      
        
    def _fc(self, x, out_dim, input_q=None, output_q=None, name="fc"):
        b, in_dim = x.get_shape().as_list()
        x = utils._fc(x, out_dim, input_q, output_q, name)
        f = 2 * (in_dim + 1) * out_dim
        w = (in_dim + 1) * out_dim
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x
    
    
    def _conv(self, x, filter_size, out_channel, stride, pad="SAME", input_q=None, output_q=None, name="conv"):
        b, h, w, in_channel = x.get_shape().as_list()
        x = utils._conv(x, filter_size, out_channel, stride, pad, input_q, output_q, name)
        f = 2 * (h/stride) * (w/stride) * in_channel * out_channel * filter_size * filter_size
        w = in_channel * out_channel * filter_size * filter_size
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x
    
