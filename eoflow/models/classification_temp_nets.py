import tensorflow as tf
from marshmallow import fields
from marshmallow.validate import OneOf

from tensorflow.keras.layers import SimpleRNN, LSTM, GRU
from tensorflow.python.keras.utils.layer_utils import print_summary

from .layers import ResidualBlock
from .classification_base import BaseClassificationModel

from . import transformer_encoder_layers
from . import pse_tae_layers


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
        output_slice_index = int(net.shape.as_list()[1] / 2) \
            if self.config.padding.lower() == 'same' else -1
        lambda_layer = tf.keras.layers.Lambda(lambda tt: tt[:, output_slice_index, :])

        if self.config.use_skip_connections:
            net = tf.keras.layers.add(skip_connections)

        if not self.config.return_sequences:
            net = lambda_layer(net)

        net = tf.keras.layers.Dense(self.config.n_classes)(net)

        net = tf.keras.layers.Softmax()(net)

        self.net = tf.keras.Model(inputs=x, outputs=net)

        print_summary(self.net)

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

        print_summary(self.net)

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

        layer_norm = fields.Bool(missing=True, description='Whether to apply layer normalization in the encoder.')
        batch_norm = fields.Bool(missing=False, description='Whether to use batch normalisation.')

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
        layers = []
        if self.config.layer_norm:
            layer_norm = tf.keras.layers.LayerNormalization()
            layers.append(layer_norm)

        # RNN layers
        layers.extend([self._rnn_layer() for _ in range(self.config.rnn_blocks-1)])
        layers.append(self._rnn_layer(last=True))

        if self.config.batch_norm:
            batch_norm = tf.keras.layers.BatchNormalization()
            layers.append(batch_norm)

        if self.config.layer_norm:
            layer_norm = tf.keras.layers.LayerNormalization()
            layers.append(layer_norm)

        dense = tf.keras.layers.Dense(units=self.config.n_classes,
                                      activation=self.config.activation,
                                      kernel_initializer=self.config.kernel_initializer,
                                      kernel_regularizer=tf.keras.regularizers.l2(self.config.kernel_regularizer))

        softmax = tf.keras.layers.Softmax()

        layers.append(dense)
        layers.append(softmax)

        self.net = tf.keras.Sequential(layers)

    def build(self, inputs_shape):
        self.net.build(inputs_shape)

        print_summary(self.net)

    def call(self, inputs, training=None):
        return self.net(inputs, training)


class TransformerEncoder(BaseClassificationModel):
    """ Implementation of a self-attention classifier

    Code is based on the Pytorch implementation of Marc Russwurm https://github.com/MarcCoru/crop-type-mapping
    """

    class TransformerEncoderSchema(BaseClassificationModel._Schema):
        keep_prob = fields.Float(required=True, description='Keep probability used in dropout layers.', example=0.5)

        num_heads = fields.Int(missing=8, description='Number of Attention heads.')
        num_layers = fields.Int(missing=4, description='Number of encoder layers.')
        num_dff = fields.Int(missing=512, description='Number of feed-forward neurons in point-wise MLP.')
        d_model = fields.Int(missing=128, description='Depth of model.')
        max_pos_enc = fields.Int(missing=24, description='Maximum length of positional encoding.')
        layer_norm = fields.Bool(missing=True, description='Whether to apply layer normalization in the encoder.')

        activation = fields.Str(missing='linear', description='Activation function used in final dense filters.')

    def init_model(self):

        self.encoder = transformer_encoder_layers.Encoder(
            num_layers=self.config.num_layers,
            d_model=self.config.d_model,
            num_heads=self.config.num_heads,
            dff=self.config.num_dff,
            maximum_position_encoding=self.config.max_pos_enc,
            layer_norm=self.config.layer_norm)

        self.dense = tf.keras.layers.Dense(units=self.config.n_classes,
                                           activation=self.config.activation)

    def build(self, inputs_shape):
        """ Build Transformer encoder architecture

        The `inputs_shape` argument is a `(N, T, D)` tuple where `N` denotes the number of samples, `T` the number of
        time-frames, and `D` the number of channels
        """
        seq_len = inputs_shape[1]

        self.net = tf.keras.Sequential([
            self.encoder,
            self.dense,
            tf.keras.layers.MaxPool1D(pool_size=seq_len),
            tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(x, axis=-2), name='squeeze'),
            tf.keras.layers.Softmax()
        ])
        # Build the model, so we can print the summary
        self.net.build(inputs_shape)

        print_summary(self.net)

    def call(self, inputs, training=None, mask=None):
        return self.net(inputs, training, mask)


class PseTae(BaseClassificationModel):
    """ Implementation of the Pixel-Set encoder + Temporal Attention Encoder sequence classifier

    Code is based on the Pytorch implementation of V. Sainte Fare Garnot et al. https://github.com/VSainteuf/pytorch-psetae
    """

    class PseTaeSchema(BaseClassificationModel._Schema):
        mlp1 = fields.List(fields.Int, missing=[10, 32, 64], description='Number of units for each layer in mlp1.')
        pooling = fields.Str(missing='mean_std', description='Methods used for pooling. Seperated by underscore. (mean, std, max, min)')
        mlp2 = fields.List(fields.Int, missing=[132, 128], description='Number of units for each layer in mlp2.')

        num_heads = fields.Int(missing=4, description='Number of Attention heads.')
        num_dff = fields.Int(missing=32, description='Number of feed-forward neurons in point-wise MLP.')
        d_model = fields.Int(missing=None, description='Depth of model.')
        mlp3 = fields.List(fields.Int, missing=[512, 128, 128], description='Number of units for each layer in mlp3.')
        dropout = fields.Float(missing=0.2, description='Dropout rate for attention encoder.')
        T = fields.Float(missing=1000, description='Number of features for attention.')
        len_max_seq = fields.Int(missing=24, description='Number of features for attention.')
        mlp4 = fields.List(fields.Int, missing=[128, 64, 32], description='Number of units for each layer in mlp4. Last layer with n_classes is added automatically.')

    def init_model(self):
        # TODO: missing features from original PseTae:
        #   * spatial encoder extra features (hand-made)
        #   * spatial encoder masking

        self.spatial_encoder = pse_tae_layers.PixelSetEncoder(
            mlp1=self.config.mlp1,
            mlp2=self.config.mlp2,
            pooling=self.config.pooling)

        self.temporal_encoder = pse_tae_layers.TemporalAttentionEncoder(
            n_head=self.config.num_heads,
            d_k=self.config.num_dff,
            d_model=self.config.d_model,
            n_neurons=self.config.mlp3,
            dropout=self.config.dropout,
            T=self.config.T,
            len_max_seq=self.config.len_max_seq)

        mlp4_layers = [pse_tae_layers.LinearLayer(out_dim) for out_dim in self.config.mlp4]
        # Final layer (logits)
        mlp4_layers.append(pse_tae_layers.LinearLayer(self.config.n_classes, batch_norm=False, activation=False))

        self.mlp4 = tf.keras.Sequential(mlp4_layers)

    def call(self, inputs, training=None, mask=None):

        out = self.spatial_encoder(inputs, training=training, mask=mask)
        out = self.temporal_encoder(out, training=training, mask=mask)
        out = self.mlp4(out, training=training, mask=mask)

        return out
