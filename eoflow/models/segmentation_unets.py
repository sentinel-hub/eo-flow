import logging
import tensorflow as tf
from marshmallow import fields

from .layers import Conv2D, Deconv2D, CropAndConcat, Conv3D, MaxPool3D, Reduce3DTo2D
from .segmentation_base import BaseSegmentationModel

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class FCNModel(BaseSegmentationModel):
    """ Implementation of a vanilla Fully-Convolutional-Network (aka U-net) """

    class FCNModelSchema(BaseSegmentationModel._Schema):
        n_layers = fields.Int(required=True, description='Number of layers of the FCN model', example=10)
        keep_prob = fields.Float(required=True, description='Keep probability used in dropout layers.', example=0.5)
        features_root = fields.Int(required=True, description='Number of features at the root level.', example=32)

        conv_size = fields.Int(missing=3, description='Size of the convolution kernels.')
        deconv_size = fields.Int(missing=2, description='Size of the deconvolution kernels.')
        conv_stride = fields.Int(missing=1, description='Stride used in convolutions.')
        dilation_rate = fields.List(fields.Int, missing=1, description='Dilation rate used in convolutions.')
        add_dropout = fields.Bool(missing=False, description='Add dropout to layers.')
        add_batch_norm = fields.Bool(missing=True, description='Add batch normalization to layers.')
        bias_init = fields.Float(missing=0.0, description='Bias initialization value.')
        use_bias = fields.Bool(missing=True, description='Add bias parameters to convolutional layer.')
        padding = fields.String(missing='VALID', description='Padding type used in convolutions.')

        pool_size = fields.Int(missing=2, description='Kernel size used in max pooling.')
        pool_stride = fields.Int(missing=2, description='Stride used in max pooling.')

        class_weights = fields.List(fields.Float, missing=None, description='Class weights used in training.')

    def build(self, inputs_shape):
        """Builds the net for input x."""

        x = tf.keras.layers.Input(inputs_shape[1:])
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
                dilation=self.config.dilation_rate,
                add_dropout=self.config.add_dropout,
                dropout_rate=dropout_rate,
                batch_normalization=self.config.add_batch_norm,
                padding=self.config.padding,
                use_bias=self.config.use_bias,
                num_repetitions=2)(net)

            connection_outputs.append(conv)

            # max pooling operation
            net = tf.keras.layers.MaxPool2D(
                pool_size=self.config.pool_size,
                strides=self.config.pool_stride,
                padding='SAME')(conv)

        # bank of 2 convolutional filters at bottom of U-net.
        bottom = Conv2D(
            filters=2 ** self.config.n_layers * self.config.features_root,
            kernel_size=self.config.conv_size,
            strides=self.config.conv_stride,
            dilation=self.config.dilation_rate,
            add_dropout=self.config.add_dropout,
            dropout_rate=dropout_rate,
            batch_normalization=self.config.add_batch_norm,
            use_bias=self.config.use_bias,
            num_repetitions=2,
            padding=self.config.padding)(net)

        net = bottom
        # Decoding path
        # the decoding path mirrors the encoding path in terms of number of features per convolutional filter
        for layer in range(self.config.n_layers):
            # find corresponding level in decoding branch
            conterpart_layer = self.config.n_layers - 1 - layer
            # get same number of features as counterpart layer
            features = 2 ** conterpart_layer * self.config.features_root

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
                dilation=self.config.dilation_rate,
                add_dropout=self.config.add_dropout,
                dropout_rate=dropout_rate,
                batch_normalization=self.config.add_batch_norm,
                use_bias=self.config.use_bias,
                num_repetitions=2,
                padding=self.config.padding)(cc)

        # final 1x1 convolution corresponding to pixel-wise linear combination of feature channels
        logits = tf.keras.layers.Conv2D(
                filters=self.config.n_classes,
                kernel_size=1)(net)

        logits = tf.keras.layers.Softmax()(logits)

        self.net = tf.keras.Model(inputs=x, outputs=logits)

    def call(self, inputs, training=None):
        return self.net(inputs, training)


class TFCNModel(BaseSegmentationModel):
    """ Implementation of a Temporal Fully-Convolutional-Network """

    class TFCNModelSchema(BaseSegmentationModel._Schema):
        n_layers = fields.Int(required=True, description='Number of layers of the FCN model', example=10)
        keep_prob = fields.Float(required=True, description='Keep probability used in dropout layers.', example=0.5)
        features_root = fields.Int(required=True, description='Number of features at the root level.', example=32)

        conv_size = fields.Int(missing=3, description='Size of the convolution kernels.')
        deconv_size = fields.Int(missing=2, description='Size of the deconvolution kernels.')
        conv_size_reduce = fields.Int(missing=3, description='Size of the kernel for time reduction.')
        conv_stride = fields.Int(missing=1, description='Stride used in convolutions.')
        add_dropout = fields.Bool(missing=False, description='Add dropout to layers.')
        add_batch_norm = fields.Bool(missing=True, description='Add batch normalization to layers.')
        bias_init = fields.Float(missing=0.0, description='Bias initialization value.')
        use_bias = fields.Bool(missing=True, description='Add bias parameters to convolutional layer.')
        padding = fields.String(missing='VALID', description='Padding type used in convolutions.')
        single_encoding_conv = fields.Bool(missing=False, description="Whether to apply 1 or 2 banks of conv filters.")

        pool_size = fields.Int(missing=2, description='Kernel size used in max pooling.')
        pool_stride = fields.Int(missing=2, description='Stride used in max pooling.')
        pool_time = fields.Bool(missing=False, description='Operate pooling over time dimension.')

        class_weights = fields.List(fields.Float, missing=None, description='Class weights used in training.')

    def build(self, inputs_shape):

        x = tf.keras.layers.Input(inputs_shape[1:])
        dropout_rate = 1 - self.config.keep_prob

        num_repetitions = 1 if self.config.single_encoding_conv else 2

        # encoding path
        net = x
        connection_outputs = []
        for layer in range(self.config.n_layers):
            # compute number of features as a function of network depth level
            features = 2 ** layer * self.config.features_root
            # bank of one 3d convolutional filter; convolution is done along the temporal as well as spatial directions
            conv = Conv3D(
                features,
                kernel_size=self.config.conv_size,
                strides=self.config.conv_stride,
                add_dropout=self.config.add_dropout,
                dropout_rate=dropout_rate,
                batch_normalization=self.config.add_batch_norm,
                num_repetitions=num_repetitions,
                use_bias=self.config.use_bias,
                padding=self.config.padding)(net)

            connection_outputs.append(conv)
            # max pooling operation
            net = MaxPool3D(
                kernel_size=self.config.pool_size,
                strides=self.config.pool_stride,
                pool_time=self.config.pool_time)(conv)

        # Bank of 1 3d convolutional filter at bottom of FCN
        bottom = Conv3D(
            2 ** self.config.n_layers * self.config.features_root,
            kernel_size=self.config.conv_size,
            strides=self.config.conv_stride,
            add_dropout=self.config.add_dropout,
            dropout_rate=dropout_rate,
            batch_normalization=self.config.add_batch_norm,
            num_repetitions=num_repetitions,
            padding=self.config.padding,
            use_bias=self.config.use_bias,
            convolve_time=(not self.config.pool_time))(net)

        # Reduce temporal dimension
        bottom = Reduce3DTo2D(
            2 ** self.config.n_layers * self.config.features_root,
            kernel_size=self.config.conv_size_reduce,
            stride=self.config.conv_stride,
            add_dropout=self.config.add_dropout,
            dropout_rate=dropout_rate)(bottom)

        net = bottom
        # decoding path
        for layer in range(self.config.n_layers):
            # find corresponding level in decoding branch
            conterpart_layer = self.config.n_layers - 1 - layer
            # get same number of features as counterpart layer
            features = 2 ** conterpart_layer * self.config.features_root

            # transposed convolution to upsample tensors
            deconv = Deconv2D(
                filters=features,
                kernel_size=self.config.deconv_size,
                batch_normalization=self.config.add_batch_norm)(net)

            # skip connection with linear combination along time
            reduced = Reduce3DTo2D(
                features,
                kernel_size=self.config.conv_size_reduce,
                stride=self.config.conv_stride,
                add_dropout=self.config.add_dropout,
                dropout_rate=dropout_rate)(connection_outputs[conterpart_layer])

            # crop and concatenate
            cc = CropAndConcat()(reduced, deconv)

            # bank of 2 convolutional layers as in standard FCN
            net = Conv2D(
                features,
                kernel_size=self.config.conv_size,
                strides=self.config.conv_stride,
                add_dropout=self.config.add_dropout,
                dropout_rate=dropout_rate,
                batch_normalization=self.config.add_batch_norm,
                padding=self.config.padding,
                use_bias=self.config.use_bias,
                num_repetitions=2)(cc)

        # final 1x1 convolution corresponding to pixel-wise linear combination of feature channels
        logits = tf.keras.layers.Conv2D(
                filters=self.config.n_classes,
                kernel_size=1)(net)

        logits = tf.keras.layers.Softmax()(logits)

        self.net = tf.keras.Model(inputs=x, outputs=logits)

    def call(self, inputs, training=None):
        return self.net(inputs, training)
