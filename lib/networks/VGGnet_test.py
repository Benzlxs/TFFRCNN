import tensorflow as tf
from network import Network
from ..fast_rcnn.config import cfg

slim = tf.contrib.slim



class VGGnet_test(Network):

    def vgg_arg_scope(self, weight_decay=0.0005):
        """Defines the VGG arg scope.

        Args:
          weight_decay: The l2 regularization coefficient.

        Returns:
          An arg_scope.
        """
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(
                                weight_decay),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc


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

        # n_classes = 21
        BENZ_num_anchors=cfg.ANCHOR_NUM
        n_classes = cfg.NCLASSES
        # anchor_scales = [8, 16, 32]
        anchor_scales = cfg.ANCHOR_SCALES
        _feat_stride = [4, ]

        #(self.feed('data')
        #     .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
        #     .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
        #     .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
        #     .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
        #     .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
        #     .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
        #     .conv(3, 3, 256, 1, 1, name='conv3_1')
        #     .conv(3, 3, 256, 1, 1, name='conv3_2')
        #     .conv(3, 3, 256, 1, 1, name='conv3_3')
        #     .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
        #     .conv(3, 3, 512, 1, 1, name='conv4_1')
        #     .conv(3, 3, 512, 1, 1, name='conv4_2')
        #     .conv(3, 3, 512, 1, 1, name='conv4_3')
        #     .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
        #     .conv(3, 3, 512, 1, 1, name='conv5_1')
        #     .conv(3, 3, 512, 1, 1, name='conv5_2')
        #     .conv(3, 3, 512, 1, 1, name='conv5_3'))
        self.layers['conv5_3'] = self.build(self.data, True)
        #========= RPN ============
        (self.feed('conv5_3')
         .conv(3, 3, 512, 1, 1, name='rpn_conv/3x3')
         .conv(1, 1, BENZ_num_anchors * 2, 1, 1, padding='VALID', relu=False, name='rpn_cls_score'))   ##benz

        (self.feed('rpn_conv/3x3')
         .conv(1, 1, BENZ_num_anchors * 4, 1, 1, padding='VALID', relu=False, name='rpn_bbox_pred'))  ## benz,  len(anchor_scales)

        #  shape is (1, H, W, Ax2) -> (1, H, WxA, 2)
        (self.feed('rpn_cls_score')
         .spatial_reshape_layer(2, name='rpn_cls_score_reshape')
         .spatial_softmax(name='rpn_cls_prob'))

        # shape is (1, H, WxA, 2) -> (1, H, W, Ax2)
        (self.feed('rpn_cls_prob')
         .spatial_reshape_layer( BENZ_num_anchors * 2, name='rpn_cls_prob_reshape'))   ## benz

        (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
         .proposal_layer(_feat_stride, anchor_scales, 'TEST', name='rois'))

        (self.feed('conv5_3', 'rois')
         .roi_pool(7, 7, 1.0 / 4, name='pool_5')
         .fc(4096, name='fc6')
         .fc(4096, name='fc7')
         .fc(n_classes, relu=False, name='cls_score')
         .softmax(name='cls_prob'))

        (self.feed('fc7')
         .fc(n_classes * 4, relu=False, name='bbox_pred'))



    def build(self,
              inputs,
              # input_pixel_size,
              is_training,
              scope='vgg_16'):   # benz, img_vgg_pyr
        """ Modified VGG for image feature extraction with pyramid features.

        Args:
            inputs: a tensor of size [batch_size, height, width, channels].
            input_pixel_size: size of the input (H x W)
            is_training: True for training, False for validation/testing.
            scope: Optional scope for the variables.

        Returns:
            The last op containing the log predictions and end_points dict.
        """
        # vgg_config = self.config


        with slim.arg_scope(self.vgg_arg_scope(
                weight_decay=   0.0005)):
            with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:  # benz, img_vgg_pyr
                end_points_collection = sc.name + '_end_points'

                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d,
                                     slim.fully_connected,
                                     slim.max_pool2d],
                                    outputs_collections=end_points_collection):

                    # Encoder
                    conv1 = slim.repeat(inputs,
                                        2,
                                        slim.conv2d,
                                        64,
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},#benz
                                        scope='conv1')
                    pool1 = slim.max_pool2d(
                        conv1, [2, 2], scope='pool1')

                    conv2 = slim.repeat(pool1,
                                        2,
                                        slim.conv2d,
                                        128,
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv2')
                    pool2 = slim.max_pool2d(
                        conv2, [2, 2], scope='pool2')

                    conv3 = slim.repeat(pool2,
                                        3,
                                        slim.conv2d,
                                        256,
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv3')
                    pool3 = slim.max_pool2d(
                        conv3, [2, 2], scope='pool3')

                    conv4 = slim.repeat(pool3,
                                        3,
                                        slim.conv2d,
                                        512,
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv4')


                    pool4 = slim.max_pool2d(
                        conv4, [2, 2], scope='pool4')

                    conv5 = slim.repeat(pool4,
                                        3,
                                        slim.conv2d,
                                        512,
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv5')




                    # Decoder (upsample and fuse features)
                    upconv4 = slim.conv2d_transpose(
                        conv5,
                        512,
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv4')

                    concat4 = tf.concat(
                        (conv4, upconv4), axis=3, name='concat4')
                    pyramid_fusion4 = slim.conv2d(
                        concat4,
                        512,
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion4')



                    upconv3 = slim.conv2d_transpose(
                        pyramid_fusion4,
                        256,
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv3')

                    concat3 = tf.concat(
                        (conv3, upconv3), axis=3, name='concat3')
                    pyramid_fusion3 = slim.conv2d(
                        concat3,
                        256,
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion3')

                    #upconv2 = slim.conv2d_transpose(
                    #    pyramid_fusion3,
                    #    vgg_config.vgg_conv2[1],
                    #    [3, 3],
                    #    stride=2,
                    #    normalizer_fn=slim.batch_norm,
                    #    normalizer_params={
                    #        'is_training': is_training},
                    #    scope='upconv2')

                    #concat2 = tf.concat(
                    #    (conv2, upconv2), axis=3, name='concat2')
                    #pyramid_fusion_2 = slim.conv2d(
                    #    concat2,
                    #    vgg_config.vgg_conv1[1],
                    #    [3, 3],
                    #    normalizer_fn=slim.batch_norm,
                    #    normalizer_params={
                    #        'is_training': is_training},
                    #    scope='pyramid_fusion2')

                    #upconv1 = slim.conv2d_transpose(
                    #    pyramid_fusion_2,
                    #    vgg_config.vgg_conv1[1],
                    #    [3, 3],
                    #    stride=2,
                    #    normalizer_fn=slim.batch_norm,
                    #    normalizer_params={
                    #        'is_training': is_training},
                    #    scope='upconv1')

                    #concat1 = tf.concat(
                    #    (conv1, upconv1), axis=3, name='concat1')
                    #pyramid_fusion1 = slim.conv2d(
                    #    concat1,
                    #    vgg_config.vgg_conv1[1],
                    #    [3, 3],
                    #    normalizer_fn=slim.batch_norm,
                    #    normalizer_params={
                    #        'is_training': is_training},
                    #    scope='pyramid_fusion1')

                feature_maps_out = pyramid_fusion3 #pyramid_fusion1

                # Convert end_points_collection into a end_point dict.
                #end_points = slim.utils.convert_collection_to_dict(
                #    end_points_collection)

                return feature_maps_out #, end_points
