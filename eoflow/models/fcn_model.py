import logging
import tensorflow as tf
import numpy as np
from marshmallow import Schema, fields

from eoflow.base import BaseModel, ModelMode
from .layers import conv1d, conv2d, deconv2d, crop_and_concat, max_pool_2d, weighted_cross_entropy

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class FCNModel(BaseModel):
    """ Implementation of a vanilla Fully-Convolutional-Network (aka U-net) """

    class FCNModelSchema(Schema):
        pass

    def _net(self, x, is_training):
        """Builds the net for input x."""

        net = x
        keep_prob = self.config.keep_prob if is_training else 1.0

        # Encoding path
        # the number of features of the convolutional kernels is proportional to the square of the level
        # for instance, starting with 32 features at the first level (layer=0), there will be 64 features at layer=1 and
        # 128 features at layer=2
        connection_outputs = []
        for layer in range(self.config.n_layers):
            # compute number of features as a function of network depth level
            features = 2 ** layer * self.config.features_root
            
            # bank of two convolutional filters
            conv = conv2d(net,
                          features,
                          is_training=is_training,
                          k_size=self.config.conv_size,
                          im_stride=self.config.conv_stride,
                          scope='encoding_' + str(layer),
                          add_dropout=self.config.add_dropout,
                          keep_prob=keep_prob,
                          add_bn=self.config.add_batch_norm,
                          bias_init=self.config.bias_init,
                          padding=self.config.padding)
            connection_outputs.append(conv)
            # max pooling operation
            net = max_pool_2d(conv,
                               ksize=self.config.pool_size,
                               stride=self.config.pool_stride)
        # bank of 2 convolutional filters at bottom of U-net.
        bottom = conv2d(net,
                        2 ** self.config.n_layers * self.config.features_root,
                        is_training=is_training,
                        k_size=self.config.conv_size,
                        im_stride=self.config.conv_stride,
                        scope='bottom',
                        add_dropout=self.config.add_dropout,
                        keep_prob=keep_prob,
                        add_bn=self.config.add_batch_norm)

        net = bottom
        # Decoding path
        # the decoding path mirrors the encoding path in terms of number of features per convolutional filter
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
            # skip connections to concatenate features from encoding path
            cc = crop_and_concat(connection_outputs[conterpart_layer],
                                 deconv)
            # bank of 2 convolutional filters
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

        if mode == ModelMode.TRAIN:
            # flatten tensors to apply class weighting
            flat_logits = tf.reshape(logits, [-1, self.config.n_classes])
            flat_labels = tf.reshape(labels, [-1, self.config.n_classes])

            if self.config.class_weights or self.config.class_weights is not None:
                loss = weighted_cross_entropy(flat_logits, flat_labels, self.config.class_weights)
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels))

            # update operations for batch-normalisation and define train stepo as minimisation of loss
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
                train_op = optimizer.minimize(loss,
                                              global_step=self.global_step_tensor)

            return train_op, loss, self.get_merged_summaries()


        # softmax to convert activations to pseudo-probabilities
        probs = tf.nn.softmax(logits, name=self.config.node_names)
        # class prediction as argmax of softmax
        preds = tf.argmax(probs, 3)

        if mode == ModelMode.PREDICT:
            
            predictions = {
                'probabilities': probs,
                'predictions': preds
            }

            return predictions

        if mode == ModelMode.EVALUATE:
            # compute classification accuracy
            accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, tf.argmax(labels, 3)), tf.float32))

            return accuracy 
