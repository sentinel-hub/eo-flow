import logging
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from marshmallow import Schema, fields

from ..base import BaseModel
from .layers import Conv2D, Deconv2D, CropAndConcat
from tensorflow.python.keras.engine import training_utils

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


def crop_loss(loss_fn):
    """ Wrapper loss. Crops labels to fit logits before applying the loss fn. """
    def _loss_fn(labels, logits):
        logits_shape = tf.shape(logits)
        labels_crop = tf.image.resize_with_crop_or_pad(labels, logits_shape[1], logits_shape[2])

        return loss_fn(labels_crop, logits)

    return _loss_fn


class FCNModel(BaseModel):
    """ Implementation of a vanilla Fully-Convolutional-Network (aka U-net) """

    class FCNModelSchema(Schema):
        learning_rate = fields.Float(missing=None, description='Learning rate used in training.', example=0.01)
        n_layers = fields.Int(required=True, description='Number of layers of the FCN model', example=10)
        n_classes = fields.Int(required=True, description='Number of classes', example=2)
        keep_prob = fields.Float(required=True, description='Keep probability used in dropout layers.', example=0.5)
        features_root = fields.Int(required=True, description='Number of features at the root level.', example=32)

        conv_size = fields.Int(missing=3, description='Size of the convolution kernels.')
        deconv_size = fields.Int(missing=2, description='Size of the deconvolution kernels.')
        conv_stride = fields.Int(missing=1, description='Stride used in convolutions.')
        add_dropout = fields.Bool(missing=False, description='Add dropout to layers.')
        add_batch_norm = fields.Bool(missing=True, description='Add batch normalization to layers.')
        bias_init = fields.Float(missing=0.0, description='Bias initialization value.')
        padding = fields.String(missing='VALID', description='Padding type used in convolutions.')

        pool_size = fields.Int(missing=2, description='Kernel size used in max pooling.')
        pool_stride = fields.Int(missing=2, description='Stride used in max pooling.')

        class_weights = fields.List(fields.Float, missing=None, description='Class weights used in training.')

        image_summaries = fields.Bool(missing=False, description='Record images summaries.')

    def build(self, inputs_shape):
        """Builds the net for input x."""

        x = layers.Input(inputs_shape[1:])
        dropout_rate = 1 - self.config.keep_prob

        # Encoding path
        # the number of features of the convolutional kernels is proportional to the square of the level
        # for instance, starting with 32 features at the first level (layer=0), there will be 64 features at layer=1 and
        # 128 features at layer=2
        net = x
        connection_outputs = []
        for layer in range(self.config.n_layers):
            # compute number of features as a function of network depth level
            features = 2 ** layer * self.config.features_root

            # bank of two convolutional filters
            conv = Conv2D(
                filters=features,
                kernel_size=self.config.conv_size,
                strides=self.config.conv_stride,
                add_dropout=self.config.add_dropout,
                dropout_rate=dropout_rate,
                batch_normalization=self.config.add_batch_norm,
                padding=self.config.padding,
                num_repetitions=2)(net)

            connection_outputs.append(conv)

            # max pooling operation
            net = layers.MaxPool2D(
                pool_size=self.config.pool_size,
                strides=self.config.pool_stride,
                padding='SAME')(conv)

        # bank of 2 convolutional filters at bottom of U-net.
        bottom = Conv2D(
            filters=2 ** self.config.n_layers * self.config.features_root,
            kernel_size=self.config.conv_size,
            strides=self.config.conv_stride,
            add_dropout=self.config.add_dropout,
            dropout_rate=dropout_rate,
            batch_normalization=self.config.add_batch_norm,
            num_repetitions=2)(net)

        self.connection_outputs = connection_outputs
        net = bottom
        # Decoding path
        # the decoding path mirrors the encoding path in terms of number of features per convolutional filter
        for layer in range(self.config.n_layers):
            # find corresponding level in decoding branch
            conterpart_layer = self.config.n_layers - 1 - layer
            # get same number of features as counterpart layer
            features = 2 ** conterpart_layer * self.config.features_root

            # transposed convolution to upsample tensors
            # shape = net.get_shape().as_list()
            # deconv_output_shape = [tf.shape(net)[0],
            #                        shape[1] * self.config.deconv_size,
            #                        shape[2] * self.config.deconv_size,
            #                        features]

            deconv = Deconv2D(
                filters=features,
                kernel_size=self.config.deconv_size,
                batch_normalization=self.config.add_batch_norm)(net)


            # # skip connections to concatenate features from encoding path
            cc = CropAndConcat()(connection_outputs[conterpart_layer],
                                 deconv)

            # bank of 2 convolutional filters
            net = Conv2D(
                filters=features,
                kernel_size=self.config.conv_size,
                strides=self.config.conv_stride,
                add_dropout=self.config.add_dropout,
                dropout_rate=dropout_rate,
                batch_normalization=self.config.add_batch_norm,
                num_repetitions=2,
                padding=self.config.padding)(cc)

        # final 1x1 convolution corresponding to pixel-wise linear combination of feature channels
        logits = layers.Conv2D(
                filters=self.config.n_classes,
                kernel_size=1)(net)

        self.net = tf.keras.Model(inputs=x, outputs=logits)


    def call(self, inputs, training=None):
        return self.net(inputs, training)

    def compile(self, **kwargs):
        # Override the compile method to wrap the loss

        if 'loss' in kwargs:
            loss = kwargs['loss']
            loss_fn = training_utils.get_loss_function(loss)

            # Wrapp loss function
            wrapped_loss_fn = crop_loss(loss_fn)
            kwargs['loss'] = wrapped_loss_fn

        super().compile(**kwargs)
