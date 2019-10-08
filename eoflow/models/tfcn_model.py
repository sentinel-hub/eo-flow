import logging
import tensorflow as tf
from marshmallow import Schema, fields
from marshmallow.validate import OneOf

from ..base import BaseModel, ModelMode
from .layers import conv1d, conv2d, conv3d, deconv2d, crop_and_concat, max_pool_3d, reduce_3d_to_2d, \
    weighted_cross_entropy,  compute_iou_loss


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class TFCNModel(BaseModel):
    """ Implementation of a Temporal Fully-Convolutional-Network """

    class TFCNModelSchema(Schema):
        learning_rate = fields.Float(missing=None, description='Learning rate used in training.', example=0.01)
        n_layers = fields.Int(required=True, description='Number of layers of the FCN model', example=10)
        n_classes = fields.Int(required=True, description='Number of classes', example=2)
        keep_prob = fields.Float(required=True, description='Keep probability used in dropout layers.', example=0.5)
        features_root = fields.Int(required=True, description='Number of features at the root level.', example=32)

        conv_size = fields.Int(missing=3, description='Size of the convolution kernels.')
        deconv_size = fields.Int(missing=2, description='Size of the deconvolution kernels.')
        conv_size_reduce = fields.Int(missing=3, description='Size of the kernel for time reduction.')
        conv_stride = fields.Int(missing=1, description='Stride used in convolutions.')
        add_dropout = fields.Bool(missing=False, description='Add dropout to layers.')
        add_batch_norm = fields.Bool(missing=True, description='Add batch normalization to layers.')
        bias_init = fields.Float(missing=0.0, description='Bias initialization value.')
        padding = fields.String(missing='VALID', description='Padding type used in convolutions.')
        single_encoding_conv = fields.Bool(missing=False, description="Whether to apply 1 or 2 banks of conv filters.") # TODO: description

        pool_size = fields.Int(missing=2, description='Kernel size used in max pooling.')
        pool_stride = fields.Int(missing=2, description='Stride used in max pooling.')
        pool_time = fields.Bool(missing=False, description='Operate pooling over time dimension.')

        class_weights = fields.List(fields.Float, missing=None, description='Class weights used in training.')
        loss = fields.String(missing='cross-entropy', description='Loss function used in training.',
                             validate=OneOf(['cross-entropy', 'iou', 'combined']))

    def _net(self, x, is_training):

        net = x
        keep_prob = self.config.keep_prob if is_training else 1.0

        # encoding path
        connection_outputs = []
        for layer in range(self.config.n_layers):
            # compute number of features as a function of network depth level
            features = 2 ** layer * self.config.features_root
            # bank of one 3d convolutional filter; convolution is done along the temporal as well as spatial directions
            conv = conv3d(net,
                          features,
                          is_training=is_training,
                          k_size=self.config.conv_size,
                          im_stride=self.config.conv_stride,
                          scope='encoding_' + str(layer),
                          add_dropout=self.config.add_dropout,
                          keep_prob=keep_prob,
                          add_bn=self.config.add_batch_norm,
                          single_filter=self.config.single_encoding_conv,
                          padding=self.config.padding)
            connection_outputs.append(conv)
            # max pooling operation
            net = max_pool_3d(conv,
                              ksize=self.config.pool_size,
                              stride=self.config.pool_stride,
                              pool_time=self.config.pool_time)
        # Bank of 1 3d convolutional filter at bottom of FCN
        bottom = conv3d(net,
                        2 ** self.config.n_layers * self.config.features_root,
                        is_training=is_training,
                        k_size=self.config.conv_size,
                        im_stride=self.config.conv_stride,
                        scope='bottom',
                        add_dropout=self.config.add_dropout,
                        keep_prob=keep_prob,
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
                                 keep_prob=keep_prob,
                                 padding="VALID")

        net = bottom
        # decoding path
        for layer in range(self.config.n_layers):
            # find corresponding level in decoding branch
            conterpart_layer = self.config.n_layers - 1 - layer
            # get same number of features as counterpart layer
            features = 2 ** conterpart_layer * self.config.features_root

            # transposed convolution to upsample tensors
            shape = net.get_shape().as_list()
            deconv_output_shape = [tf.shape(net)[0],
                                   shape[1] * self.config.deconv_size,
                                   shape[2] * self.config.deconv_size,
                                   features]
            deconv = deconv2d(net,
                              deconv_output_shape,
                              k_size=self.config.deconv_size,
                              is_training=is_training,
                              scope='deconv_' + str(conterpart_layer),
                              add_bn=self.config.add_batch_norm)
            # skip connection with linear combination along time
            reduced = reduce_3d_to_2d(connection_outputs[conterpart_layer],
                                      features,
                                      k_size=self.config.conv_size_reduce,
                                      im_stride=self.config.conv_stride,
                                      add_dropout=self.config.add_dropout,
                                      keep_prob=keep_prob,
                                      scope='decoding_reduced_' + str(conterpart_layer),
                                      padding="VALID")
            # crop and concatenate
            cc = crop_and_concat(reduced, deconv)
            # bank of 2 convolutional layers as in standard FCN
            net = conv2d(cc,
                         features,
                         k_size=self.config.conv_size,
                         im_stride=self.config.conv_stride,
                         is_training=is_training,
                         scope='decoding_' + str(conterpart_layer),
                         add_dropout=self.config.add_dropout,
                         keep_prob=keep_prob,
                         add_bn=self.config.add_batch_norm,
                         padding=self.config.padding)
        # final 1x1 convolution corresponding to pixel-wise linear combination of feature channels
        logits = conv1d(net,
                        self.config.n_classes,
                        scope='logits',
                        bias_init=self.config.bias_init)

        return logits

    def build_model(self, features, labels, mode):
        x = features
        is_training = mode == ModelMode.TRAIN

        # Build net
        logits = self._net(x, is_training)

        # softmax to convert activations to pseudo-probabilities
        probs = tf.nn.softmax(logits)
        # class prediction as argmax of softmax
        preds = tf.argmax(probs[..., 1:], 3)

        if mode == ModelMode.TRAIN:
            out_shape = tf.shape(logits)
            labels_cropped = tf.image.resize_with_crop_or_pad(labels, out_shape[1], out_shape[2])

            self.add_summary(tf.summary.image('input', features[:,0,...][...,0:3]))
            self.add_summary(tf.summary.image('labels_raw', labels[...,0:3]))
            self.add_summary(tf.summary.image('labels', labels_cropped[...,0:3]))
            self.add_summary(tf.summary.image('output', logits[...,0:3]))

            # flatten tensors to apply class weighting
            flat_logits = tf.reshape(logits, [-1, self.config.n_classes])
            flat_labels = tf.reshape(labels_cropped, [-1, self.config.n_classes])

            # cross-entropy loss w or w/o class weights
            if self.config.class_weights is not None:
                cross_entropy_loss = weighted_cross_entropy(flat_logits, flat_labels, self.config.class_weights)
            else:
                cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                                            labels=flat_labels))
            # intersection over union loss
            iou_loss = compute_iou_loss(self.config.n_classes, probs, preds, tf.cast(labels_cropped, tf.float32),
                                        class_weights=self.config.class_weights, exclude_background=False)

            # Total loss, which is cross-entropy, IOU or a sum of the two
            if self.config.loss == 'cross-entropy':
                loss = cross_entropy_loss
            elif self.config.loss == 'iou':
                loss = iou_loss
            elif self.config.loss == 'combined':
                loss = cross_entropy_loss + iou_loss

            # update operations for batch-normalisation and define train stepo as minimisation of loss
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
                train_op = optimizer.minimize(loss,
                                              global_step=self.global_step_tensor)

            return train_op, loss, self.get_merged_summaries()

        if mode == ModelMode.PREDICT:

            predictions = {
                'probabilities': probs,
                'predictions': preds
            }

            return predictions

        if mode == ModelMode.EVALUATE:
            # compute classification accuracy
            accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, tf.argmax(labels[..., 1:], 3)), tf.float32))

            return accuracy
