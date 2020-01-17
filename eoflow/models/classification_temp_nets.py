import logging
import tensorflow as tf
from marshmallow import fields
from marshmallow.validate import OneOf

from tensorflow.keras.layers import SimpleRNN, LSTM, GRU

from .layers import ResidualBlock
from .classification_base import BaseClassificationModel

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

rnn_layers = dict(rnn=SimpleRNN, gru=GRU, lstm=LSTM)


class TCNModel(BaseClassificationModel):
    """ Implementation of the TCN network taken form the keras-TCN implementation

        https://github.com/philipperemy/keras-tcn
    """

    class TCNModelSchema(BaseClassificationModel._Schema):
        keep_prob = fields.Float(required=True, description='Keep probability used in dropout layers.', example=0.5)

        kernel_size = fields.Int(missing=2, description='Size of the convolution kernels.')
        nb_filters = fields.Int(missing=64, description='Number of convolutional filters.')
        nb_stacks = fields.Int(missing=1)
        dilations = fields.List(fields.Int, missing=[1, 2, 4, 8, 16, 32], description='Size of dilations used in the '
                                                                                      'covolutional layers')
        padding = fields.String(missing='CAUSAL', validate=OneOf(['CAUSAL', 'SAME']),
                                description='Padding type used in convolutions.')
        use_skip_connections = fields.Bool(missing=True, description='Flag to whether to use skip connections.')
        return_sequences = fields.Bool(missing=False, description='Flag to whether return sequences or not.')
        activation = fields.Str(missing='linear', description='Activation function used in final filters.')
        kernel_initializer = fields.Str(missing='he_normal', description='method to initialise kernel parameters.')

        use_batch_norm = fields.Bool(missing=False, description='Whether to use batch normalisation.')
        use_layer_norm = fields.Bool(missing=False, description='Whether to use layer normalisation.')

    def build(self, inputs_shape):
        """ Build TCN architecture

        The `inputs_shape` argument is a `(N, T, D)` tuple where `N` denotes the number of samples, `T` the number of
        time-frames, and `D` the number of channels
        """
        x = tf.keras.layers.Input(inputs_shape[1:])

        dropout_rate = 1 - self.config.keep_prob

        net = x

        net = tf.keras.layers.Conv1D(filters=self.config.nb_filters,
                                     kernel_size=1,
                                     padding=self.config.padding,
                                     kernel_initializer=self.config.kernel_initializer)(net)

        # list to hold all the member ResidualBlocks
        residual_blocks = list()
        skip_connections = list()

        total_num_blocks = self.config.nb_stacks * len(self.config.dilations)
        if not self.config.use_skip_connections:
            total_num_blocks += 1  # cheap way to do a false case for below

        for s in range(self.config.nb_stacks):
            for d in self.config.dilations:
                net, skip_out = ResidualBlock(dilation_rate=d,
                                              nb_filters=self.config.nb_filters,
                                              kernel_size=self.config.kernel_size,
                                              padding=self.config.padding,
                                              activation=self.config.activation,
                                              dropout_rate=dropout_rate,
                                              use_batch_norm=self.config.use_batch_norm,
                                              use_layer_norm=self.config.use_layer_norm,
                                              kernel_initializer=self.config.kernel_initializer,
                                              last_block=len(residual_blocks) + 1 == total_num_blocks,
                                              name=f'residual_block_{len(residual_blocks)}')(net)
                residual_blocks.append(net)
                skip_connections.append(skip_out)

        # Author: @karolbadowski.
        output_slice_index = int(net.output_shape.as_list()[1] / 2) \
            if self.config.padding.lower() == 'same' else -1
        lambda_layer = tf.keras.layers.Lambda(lambda tt: tt[:, output_slice_index, :])

        if self.config.use_skip_connections:
            net = tf.keras.layers.add(skip_connections)

        if not self.config.return_sequences:
            net = lambda_layer(net)

        net = tf.keras.layers.Dense(self.config.n_classes)(net)

        net = tf.keras.layers.Softmax()(net)

        self.net = tf.keras.Model(inputs=x, outputs=net)

    def call(self, inputs, training=None):
        return self.net(inputs, training)


class TempCNNModel(BaseClassificationModel):
    """ Implementation of the TempCNN network taken from the temporalCNN implementation

        https://github.com/charlotte-pel/temporalCNN
    """

    class TempCNNModelSchema(BaseClassificationModel._Schema):
        keep_prob = fields.Float(required=True, description='Keep probability used in dropout layers.', example=0.5)

        kernel_size = fields.Int(missing=5, description='Size of the convolution kernels.')
        nb_conv_filters = fields.Int(missing=16, description='Number of convolutional filters.')
        nb_conv_stacks = fields.Int(missing=3, description='Number of convolutional blocks.')
        nb_conv_strides = fields.Int(missing=1, description='Value of convolutional strides.')
        nb_fc_neurons = fields.Int(missing=256, description='Number of Fully Connect neurons.')
        nb_fc_stacks = fields.Int(missing=1, description='Number of fully connected layers.')
        padding = fields.String(missing='SAME', validate=OneOf(['SAME']),
                                description='Padding type used in convolutions.')
        activation = fields.Str(missing='relu', description='Activation function used in final filters.')
        kernel_initializer = fields.Str(missing='he_normal', description='Method to initialise kernel parameters.')
        kernel_regularizer = fields.Float(missing=1e-6, description='L2 regularization parameter.')

        use_batch_norm = fields.Bool(missing=False, description='Whether to use batch normalisation.')

    def build(self, inputs_shape):
        """ Build TCN architecture

        The `inputs_shape` argument is a `(N, T, D)` tuple where `N` denotes the number of samples, `T` the number of
        time-frames, and `D` the number of channels
        """
        x = tf.keras.layers.Input(inputs_shape[1:])

        dropout_rate = 1 - self.config.keep_prob

        net = x
        for _ in range(self.config.nb_conv_stacks):
            net = tf.keras.layers.Conv1D(filters=self.config.nb_conv_filters,
                                         kernel_size=self.config.kernel_size,
                                         strides=self.config.nb_conv_strides,
                                         padding=self.config.padding,
                                         kernel_initializer=self.config.kernel_initializer,
                                         kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net)
            if self.config.use_batch_norm:
                net = tf.keras.layers.BatchNormalization(axis=-1)(net)

            net = tf.keras.layers.Activation(self.config.activation)(net)

            net = tf.keras.layers.Dropout(dropout_rate)(net)

        net = tf.keras.layers.Flatten()(net)

        for _ in range(self.config.nb_fc_stacks):
            net = tf.keras.layers.Dense(units=self.config.nb_fc_neurons,
                                        kernel_initializer=self.config.kernel_initializer,
                                        kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net)
            if self.config.use_batch_norm:
                net = tf.keras.layers.BatchNormalization(axis=-1)(net)

            net = tf.keras.layers.Activation(self.config.activation)(net)

            net = tf.keras.layers.Dropout(dropout_rate)(net)

        net = tf.keras.layers.Dense(units=self.config.n_classes,
                                    kernel_initializer=self.config.kernel_initializer,
                                    kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))(net)

        net = tf.keras.layers.Softmax()(net)

        self.net = tf.keras.Model(inputs=x, outputs=net)

    def call(self, inputs, training=None):
        return self.net(inputs, training)


class BiRNN(BaseClassificationModel):
    """ Implementation of a Bidirectional Recurrent Neural Network

    This implementation allows users to define which RNN layer to use, e.g. SimpleRNN, GRU or LSTM
    """

    class BiRNNModelSchema(BaseClassificationModel._Schema):
        rnn_layer = fields.String(required=True, validate=OneOf(['rnn', 'lstm', 'gru']),
                                  description='Type of RNN layer to use')

        keep_prob = fields.Float(required=True, description='Keep probability used in dropout layers.', example=0.5)

        rnn_units = fields.Int(missing=64, description='Size of the convolution kernels.')
        rnn_blocks = fields.Int(missing=1, description='Number of LSTM blocks')
        bidirectional = fields.Bool(missing=True, description='Whether to use a bidirectional layer')

        activation = fields.Str(missing='linear', description='Activation function used in final dense filters.')
        kernel_initializer = fields.Str(missing='he_normal', description='Method to initialise kernel parameters.')
        kernel_regularizer = fields.Float(missing=1e-6, description='L2 regularization parameter.')


    def _rnn_layer(self, last=False):
        """ Returns a RNN layer for current configuration. Use `last=True` for the last RNN layer. """

        RNNLayer = rnn_layers[self.config.rnn_layer]
        dropout_rate = 1 - self.config.keep_prob

        layer = RNNLayer(units=self.config.rnn_units,
                         dropout=dropout_rate,
                         return_sequences=False if last else True)

        # Use bidirectional if specified
        if self.config.bidirectional:
            layer = tf.keras.layers.Bidirectional(layer)

        return layer

    def init_model(self):
        """ Creates the RNN model architecture. """

        # RNN layers
        layers = [self._rnn_layer() for _ in range(self.config.rnn_blocks-1)]
        layers.append(self._rnn_layer(last=True))

        dense = tf.keras.layers.Dense(units=self.config.n_classes,
                                      activation=self.config.activation,
                                      kernel_initializer=self.config.kernel_initializer,
                                      kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))

        softmax = tf.keras.layers.Softmax()

        layers.append(dense)
        layers.append(softmax)

        self.net = tf.keras.Sequential(layers)

    def call(self, inputs, training=None):
        return self.net(inputs, training)
