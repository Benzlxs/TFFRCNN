# --------------------------------------------------------
# TFFRCNN - Resnet50
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by miraclebiu
# --------------------------------------------------------

import tensorflow as tf
from .network import Network
from ..fast_rcnn.config import cfg
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block



class Resnet101_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name='im_info')
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='gt_boxes')
        self.gt_ishard = tf.placeholder(tf.int32, shape=[None], name='gt_ishard')
        self.dontcare_areas = tf.placeholder(tf.float32, shape=[None, 4], name='dontcare_areas')
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data':self.data, 'im_info':self.im_info, 'gt_boxes':self.gt_boxes,\
                            'gt_ishard': self.gt_ishard, 'dontcare_areas': self.dontcare_areas})
        self.trainable = trainable
        self.setup()

    def setup(self):

        n_classes = cfg.NCLASSES
        # anchor_scales = [8, 16, 32]
        anchor_scales = cfg.ANCHOR_SCALES
        _feat_stride = [8, ]
        num_anchors=cfg.ANCHOR_NUM

        self.layers['res4b22_relu'] = self.build(self.data, True)



        #========= RPN ============
        (self.feed('res4b22_relu')
             .conv(3,3,512,1,1,name='rpn_conv/3x3')
             .conv(1,1, num_anchors*2 ,1 , 1, padding='VALID', relu = False, name='rpn_cls_score'))

        (self.feed('rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info')
             .anchor_target_layer(_feat_stride, anchor_scales, name = 'rpn-data' ))
        # Loss of rpn_cls & rpn_boxes

        (self.feed('rpn_conv/3x3')
             .conv(1,1, num_anchors*4, 1, 1, padding='VALID', relu = False, name='rpn_bbox_pred'))

        #========= RoI Proposal ============
        (self.feed('rpn_cls_score')
             .spatial_reshape_layer(2, name = 'rpn_cls_score_reshape')
             .spatial_softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob')
             .spatial_reshape_layer( num_anchors*2, name = 'rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape','rpn_bbox_pred','im_info')
             .proposal_layer(_feat_stride, anchor_scales, 'TRAIN',name = 'rpn_rois'))

        (self.feed('rpn_rois','gt_boxes', 'gt_ishard', 'dontcare_areas')
             .proposal_target_layer(n_classes,name = 'roi-data'))


        #========= RCNN ============
        (self.feed('res4b22_relu','roi-data')
             .roi_pool(7,7,1.0/16,name='res5a_branch2a_roipooling')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res5a_branch2a',padding='VALID')
             .batch_normalization(relu=True, name='bn5a_branch2a',is_training=False)
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5a_branch2b')
             .batch_normalization(relu=True, name='bn5a_branch2b',is_training=False)
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c')
             .batch_normalization(name='bn5a_branch2c',is_training=False,relu=False))

        (self.feed('res5a_branch2a_roipooling')
             .conv(1, 1, 2048, 2, 2, biased=False, relu=False, name='res5a_branch1',padding='VALID')
             .batch_normalization(name='bn5a_branch1',is_training=False,relu=False))


        (self.feed('bn5a_branch1',
                   'bn5a_branch2c')
             .add(name='res5a')
             .relu(name='res5a_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a')
             .batch_normalization(relu=True, name='bn5b_branch2a',is_training=False)
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5b_branch2b')
             .batch_normalization(relu=True, name='bn5b_branch2b',is_training=False)
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c')
             .batch_normalization(name='bn5b_branch2c',is_training=False,relu=False))

        (self.feed('res5a_relu',
                   'bn5b_branch2c')
             .add(name='res5b')
             .relu(name='res5b_relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5c_branch2a')
             .batch_normalization(relu=True, name='bn5c_branch2a',is_training=False)
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, name='res5c_branch2b')
             .batch_normalization(relu=True, name='bn5c_branch2b',is_training=False)
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c')
             .batch_normalization(name='bn5c_branch2c',is_training=False,relu=False))

        (self.feed('res5b_relu',
                   'bn5c_branch2c')
             .add(name='res5c')
             .relu(name='res5c_relu')
             .avg_pool(4, 4, 1, 1, padding='VALID', name='pool5')
             .fc(1000, relu=False, name='fc1000')
             .softmax(name='prob'))


    def _build_base(self, input_img):
        with tf.variable_scope(self._scope, self._scope):
            net = resnet_utils.conv2d_same( input_img, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')

        return net




    def build(self,
              inputs,
              # input_pixel_size,
              is_training,
              scope='resnet_v1_101',
              rpn_weight_decay=0.0001 ):  ## scope is important variable to set
        """ resnet
        args:
            inputs: a tensor of size [batch_size, height, width, channels].
            input_pixel_size: size of the input (H x W)
            is_training: True for training, False for validation/testing.
            scope: Optional scope for the variables.

        Returns:
            The last op containing the log predictions and end_points dict.
        """
        # res_config = self.config
        fixed_block = cfg.fixed_block
        self._scope = scope
        rpn_weight_decay = cfg.rpn_weight_decay

        ## backbone
        with slim.arg_scope(resnet_arg_scope(is_training=False)):
            net_base = self._build_base(inputs)


        blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                  resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                  resnet_v1_block('block3', base_depth=256, num_units=23, stride=2),   ##decreasing factor is 32
                  resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]


        with slim.arg_scope(resnet_arg_scope(is_training=False)):
            net_conv1, net_dict1 = resnet_v1.resnet_v1(net_base,
                                              blocks[0:fixed_block],
                                              global_pool=False,
                                              include_root_block=False,  ## no resue
                                              scope=self._scope)

        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            net_conv2, net_dict2 = resnet_v1.resnet_v1(net_conv1,
                                              blocks[fixed_block:],
                                              global_pool=False,
                                              include_root_block=False,
                                              scope=self._scope)

        ## build feature maps
        feature_maps_dict  = {
            'C2': net_dict1['resnet_v1_101/block1/unit_2/bottleneck_v1'],
            'C3': net_dict2['resnet_v1_101/block2/unit_3/bottleneck_v1'],
            'C4': net_dict2['resnet_v1_101/block3/unit_22/bottleneck_v1'],
            'C5': net_dict2['resnet_v1_101/block4']
        }


        ## build pyramid feature maps
        feature_pyramid = {}

        upsample_method = 'deconv'   ## put this setting into configuration file later

        if upsample_method == 'deconv':
            ## using deconvolution to build pyramid feature image
            with tf.variable_scope('build_feature_pyramid'):
                with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer( rpn_weight_decay )):
                     feature_pyramid['P5'] = slim.conv2d(feature_maps_dict['C5'],
                                                        num_outputs=256,
                                                        kernel_size=[1, 1],
                                                        stride=1,
                                                        scope='build_P5')
                     ##p6 is down sample of p5
                     for layer in range(4, 1, -1):
                         p, c = feature_pyramid['P' + str(layer + 1)], feature_maps_dict['C' + str(layer)]

                         up_sample = slim.conv2d_transpose(
                                         p,
                                         num_outputs=256 ,
                                         kernel_size=[3, 3],
                                         stride=2,
                                         normalizer_fn=slim.batch_norm,
                                         normalizer_params={
                                             'is_training': is_training},
                                         scope ='build_P%d/up_sampling_deconvolution' % layer)

                         c = slim.conv2d(c,
                                         num_outputs=256,
                                         kernel_size=[1, 1],
                                         stride=1,
                                         normalizer_fn=slim.batch_norm,
                                         normalizer_params={
                                            'is_training': is_training },
                                         scope='build_P%d/reduce_dimension' % layer)

                         concat = tf.concat(
                             (c, up_sample), axis=3, name='build_P%d/concate_layer' %layer)

                         p = slim.conv2d(concat,
                                         num_outputs=256,
                                         kernel_size=[3, 3],
                                         stride=1,
                                         normalizer_fn=slim.batch_norm,
                                         normalizer_params={
                                             'is_training': is_training},
                                         scope= 'build_P%d/for_feature_pyramid' % layer)

                         feature_pyramid['P' + str(layer)] = p
        return feature_pyramid['P2']


