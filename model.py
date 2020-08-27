#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WORK: Created on Mon 12 Nov 14:13:52 2018
    Code Reproduction
    Model (ResNet, ConvLSTM): For object detection and tracking
@author: nwoye (CAMMA, iCube) Universite' de Strasbourg, France
"""

import tensorflow as tf
import numpy as np
from lib import resnet as resnet 
from lib.convlstm_cell import ConvLSTM

    
        
class Model(object):
    def __init__(self, images, seek=[1], num_classes=7): 
        self._images                = images 
        self._seek                  = seek
        self._is_training           = tf.constant(True)
        self._num_classes           = num_classes 
        self._loc_height            = 60
        self._loc_width             = 107
        self._loc_channel           = 7
        self._prev_track_loc        = {0:[0,0,0,0],1:[0,0,0,0],2:[0,0,0,0],3:[0,0,0,0],4:[0,0,0,0],5:[0,0,0,0],6:[0,0,0,0]}
        self._prev_track_center     = {0:[0,0],    1:[0,0],    2:[0,0,],   3:[0,0],    4:[0,0],    5:[0,0],    6:[0,0]    }
        self._img_batch, self._img_height, self._img_width, self._img_channel = self._input_shape = self._images.get_shape().as_list()
        
        

        '''Initialize the parameters for the model.
        Args:
            images:         4D input tensor [batch_size, height, width, channel] of image of type int32, int64, float32 or float64.
            num_classes:    Integer, representing number of classes for the detection or tracking task. Default: 7
            seek:           Integer, current video index. Use for resetting the model cell states at the beginning of each videos
            is_training:    Boolean, (optional), set True for training and False for validation and testing. Default: True
        '''   

    #%% Build model in graph   
    def build_model(self):
        ''' Build the deep learning network for tools detection and/or tracking based on the selected model
        Args: super(Model)
        Returns: 
            logits:     Float, [batch_size, num_classes] estimated tools logits which translates to tools presence probabilities after sigmoid operation.
            lhmaps:     Float, [batch_size, height, width, num_classes] channel localization map, one channel corresponding to each tool class in the order of training. 
        '''
        with tf.name_scope('Model'): 
            # feature extraction
            x =  resnet.ResNet(images=self._images, version=18, is_training=self._is_training)._build_model()
            # localization and tracking
            with tf.variable_scope('ExtraNet') as scope: 
                with tf.variable_scope('spatio-temporal'):  
                    spt_x = ConvLSTM(filters=512, kernel=3, strides=1, is_training=self._is_training, scope='convlstm')._residual(x, seek=self._seek)                  
                lhmaps  = self._locnet(x=spt_x, filters=self._num_classes, name='FCN')
                logits  = self._spatial_pooling(x=lhmaps)          
                bboxes = None
        return logits, lhmaps  
    
    
    def _spatial_pooling(self, x, name='spatial_pooling'):
        with tf.name_scope(name):
            return tf.reduce_max(x, reduction_indices=[1,2]) + 0.6*tf.reduce_min(x, reduction_indices=[1,2])
    
    
    def _locnet(self, x, kernel=1, filters=7, strides=1, name='conv6'):
        with tf.variable_scope(name) as scope:          
            x = tf.layers.conv2d(inputs=x, filters=filters, kernel_size=kernel, 
                            strides=strides, padding='valid', use_bias=True, 
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            print('\tBuilding units: {}: -> {}'.format( scope.name, x.get_shape() )) 
        return x    
    