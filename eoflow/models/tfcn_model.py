import logging

import tensorflow as tf

from .layers import conv1d, conv2d, conv3d, deconv2d, crop_and_concat, max_pool_3d, reduce_3d_to_2d, \
    weighted_cross_entropy,  compute_iou_loss
from ..base import BaseModel


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class TFCNModel(BaseModel):
    """ Implementation of a Temporal Fully-Convolutional-Network """
    def __init__(self, config):
        super(TFCNModel, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_model(self):
        logging.debug("Building model")
        # placeholder for training flag
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        # placeholder for dropout
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # placeholder for input 5d tensor
        self.x = tf.placeholder(tf.float32,
                                shape=[None,
                                       self.config.im_size[0],
                                       self.config.im_size[1],
                                       self.config.im_size[2],
                                       self.config.im_size[3]],
                                name='images')
        # placeholder for labels 4d tensor
        self.y = tf.placeholder(tf.float32,
                                shape=[None, self.config.lb_size[0], self.config.lb_size[1], self.config.n_classes],
                                name='labels')
        # encoding path
        connection_outputs = []
        for layer in range(self.config.n_layers):
            # compute number of features as a function of network depth level
            features = 2 ** layer * self.config.features_root
            if layer == 0:
                # if top layer, previous tensor is input data
                prev = self.x
            else:
                # if not top layer, previous tensor is pooling
                prev = pool
            # bank of one 3d convolutional filter; convolution is done along the temporal as well as spatial directions
            conv = conv3d(prev,
                          features,
                          is_training=self.is_training,
                          k_size=self.config.conv_size,
                          im_stride=self.config.conv_stride,
                          scope='encoding_' + str(layer),
                          add_dropout=self.config.add_dropout,
                          keep_prob=self.keep_prob,
                          add_bn=self.config.add_batch_norm,
                          single_filter=self.config.single_encoding_conv,
                          padding=self.config.padding)
            connection_outputs.append(conv)
            # max pooling operation
            pool = max_pool_3d(conv,
                               ksize=self.config.pool_size,
                               stride=self.config.pool_stride,
                               pool_time=self.config.pool_time)
        # Bank of 1 3d convolutional filter at bottom of FCN
        bottom = conv3d(pool,
                        2 ** self.config.n_layers * self.config.features_root,
                        is_training=self.is_training,
                        k_size=self.config.conv_size,
                        im_stride=self.config.conv_stride,
                        scope='bottom',
                        add_dropout=self.config.add_dropout,
                        keep_prob=self.keep_prob,
                        add_bn=self.config.add_batch_norm,
                        single_filter=self.config.single_encoding_conv,
                        padding=self.config.padding,
                        convolve_time=(not self.config.pool_time))
        # Reduce temporal dimension
        bottom = reduce_3d_to_2d(bottom,
                                 2 ** self.config.n_layers * self.config.features_root,
                                 k_size=self.config.conv_size_reduce,
                                 im_stride=self.config.conv_stride,
                                 scope='reduce_bottom',
                                 add_dropout=self.config.add_dropout,
                                 keep_prob=self.keep_prob,
                                 padding="VALID")
        # decoding path
        for layer in range(self.config.n_layers):
            # find corresponding level in decoding branch
            conterpart_layer = self.config.n_layers - 1 - layer
            # get same number of features as counterpart layer
            features = 2 ** conterpart_layer * self.config.features_root
            if layer == 0:
                prev = bottom
            else:
                prev = conv_decoding
            # transposed convolution to upsample tensors
            shape = prev.get_shape().as_list()
            deconv_output_shape = [tf.shape(prev)[0],
                                   shape[1] * self.config.deconv_size,
                                   shape[2] * self.config.deconv_size,
                                   features]
            deconv = deconv2d(prev,
                              deconv_output_shape,
                              k_size=self.config.deconv_size,
                              is_training=self.is_training,
                              scope='deconv_' + str(conterpart_layer),
                              add_bn=self.config.add_batch_norm)
            # skip connection with linear combination along time
            reduced = reduce_3d_to_2d(connection_outputs[conterpart_layer],
                                      features,
                                      k_size=self.config.conv_size_reduce,
                                      im_stride=self.config.conv_stride,
                                      add_dropout=self.config.add_dropout,
                                      keep_prob=self.keep_prob,
                                      scope='decoding_reduced_' + str(conterpart_layer),
                                      padding="VALID")
            # crop and concatenate
            cc = crop_and_concat(reduced, deconv)
            # bank of 2 convolutional layers as in standard FCN
            conv_decoding = conv2d(cc,
                                   features,
                                   k_size=self.config.conv_size,
                                   im_stride=self.config.conv_stride,
                                   is_training=self.is_training,
                                   scope='decoding_' + str(conterpart_layer),
                                   add_dropout=self.config.add_dropout,
                                   keep_prob=self.keep_prob,
                                   add_bn=self.config.add_batch_norm,
                                   padding=self.config.padding)
        # final 1x1 convolution corresponding to pixel-wise linear combination of feature channels
        logits = conv1d(conv_decoding,
                        self.config.n_classes,
                        scope='logits',
                        bias_init=self.config.bias_init)
        # softmax to convert activations to pseudo-probabilities
        self.probs = tf.nn.softmax(logits, name=self.config.node_names)
        # class prediction as argmax of softmax
        self.preds = tf.argmax(self.probs[..., 1:], 3)
        # compute classification accuracy
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.preds, tf.argmax(self.y[..., 1:], 3)), tf.float32))
        # flatten tensors to apply class weighting
        flat_logits = tf.reshape(logits, [-1, self.config.n_classes])
        flat_labels = tf.reshape(self.y, [-1, self.config.n_classes])
        # cross-entropy loss w or w/o class weights
        cross_entropy_loss = weighted_cross_entropy(flat_logits, flat_labels, self.config.class_weights) if \
            self.config.class_weights else tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                                                  labels=flat_labels))
        # intersection over union loss
        iou_loss = compute_iou_loss(self.config.n_classes, self.probs, self.preds, self.y,
                                    class_weights=self.config.class_weights, exclude_background=False)

        # Total loss, which is cross-entropy, IOU or a sum of the two
        if self.config.loss == 'cross-entropy':
            self.loss = cross_entropy_loss
        elif self.config.loss == 'iou':
            self.loss = iou_loss
        elif self.config.loss == 'combined':
            self.loss = cross_entropy_loss + iou_loss
        else:
            raise ValueError("Unknown cost function: " + self.config.loss)

        # update operations for batch-normalisation and define train stepo as minimisation of loss
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss,
                                                                                         global_step=
                                                                                         self.global_step_tensor)

    def init_saver(self):
        logging.debug("Initialising model saver")
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
