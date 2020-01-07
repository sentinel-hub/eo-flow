import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Activation, SpatialDropout1D, Lambda
from tensorflow.keras.layers import Conv1D, BatchNormalization, LayerNormalization

from .conv_cells import ConvGRUCell


class ResidualBlock(tf.keras.layers.Layer):
    """ Code taken from keras-tcn implementation on available on
    https://github.com/philipperemy/keras-tcn/blob/master/tcn/tcn.py#L140 """
    def __init__(self,
                 dilation_rate,
                 nb_filters,
                 kernel_size,
                 padding,
                 activation='relu',
                 dropout_rate=0,
                 kernel_initializer='he_normal',
                 use_batch_norm=False,
                 use_layer_norm=False,
                 last_block=True,
                 **kwargs):

        """ Defines the residual block for the WaveNet TCN

        :param dilation_rate: The dilation power of 2 we are using for this residual block
        :param nb_filters: The number of convolutional filters to use in this block
        :param kernel_size: The size of the convolutional kernel
        :param padding: The padding used in the convolutional layers, 'same' or 'causal'.
        :param activation: The final activation used in o = Activation(x + F(x))
        :param dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
        :param kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
        :param use_batch_norm: Whether to use batch normalization in the residual layers or not.
        :param use_layer_norm: Whether to use layer normalization in the residual layers or not.
        :param last_block: Whether to add a residual connection to the convolution layer or not.
        :param kwargs: Any initializers for Layer class.
        """

        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.kernel_initializer = kernel_initializer
        self.last_block = last_block
        self.residual_layers = list()
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None

        super(ResidualBlock, self).__init__(**kwargs)

    def _add_and_activate_layer(self, layer):
        """Helper function for building layer
        Args:
            layer: Appends layer to internal layer list and builds it based on the current output
                   shape of ResidualBlocK. Updates current output shape.
        """
        self.residual_layers.append(layer)
        self.residual_layers[-1].build(self.res_output_shape)
        self.res_output_shape = self.residual_layers[-1].compute_output_shape(self.res_output_shape)

    def build(self, input_shape):

        with tf.keras.backend.name_scope(self.name):  # name scope used to make sure weights get unique names
            self.res_output_shape = input_shape

            for k in range(2):
                name = f'conv1D_{k}'
                with tf.keras.backend.name_scope(name):  # name scope used to make sure weights get unique names
                    self._add_and_activate_layer(Conv1D(filters=self.nb_filters,
                                                        kernel_size=self.kernel_size,
                                                        dilation_rate=self.dilation_rate,
                                                        padding=self.padding,
                                                        name=name,
                                                        kernel_initializer=self.kernel_initializer))

                if self.use_batch_norm:
                    self._add_and_activate_layer(BatchNormalization())
                elif self.use_layer_norm:
                    self._add_and_activate_layer(LayerNormalization())

                self._add_and_activate_layer(Activation('relu'))
                self._add_and_activate_layer(SpatialDropout1D(rate=self.dropout_rate))

            if not self.last_block:
                # 1x1 conv to match the shapes (channel dimension).
                name = f'conv1D_{k+1}'
                with tf.keras.backend.name_scope(name):
                    # make and build this layer separately because it directly uses input_shape
                    self.shape_match_conv = Conv1D(filters=self.nb_filters,
                                                   kernel_size=1,
                                                   padding='same',
                                                   name=name,
                                                   kernel_initializer=self.kernel_initializer)

            else:
                self.shape_match_conv = Lambda(lambda x: x, name='identity')

            self.shape_match_conv.build(input_shape)
            self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)

            self.final_activation = Activation(self.activation)
            self.final_activation.build(self.res_output_shape)  # probably isn't necessary

            # this is done to force keras to add the layers in the list to self._layers
            for layer in self.residual_layers:
                self.__setattr__(layer.name, layer)

            super(ResidualBlock, self).build(input_shape)  # done to make sure self.built is set True

    def call(self, inputs, training=None):
        """
        Returns: A tuple where the first element is the residual model tensor, and the second
                 is the skip connection tensor.
        """
        x = inputs
        for layer in self.residual_layers:
            if isinstance(layer, SpatialDropout1D):
                x = layer(x, training=training)
            else:
                x = layer(x)

        x2 = self.shape_match_conv(inputs)
        res_x = tf.keras.layers.add([x2, x])
        return [self.final_activation(res_x), x]

    def compute_output_shape(self, input_shape):
        return [self.res_output_shape, self.res_output_shape]


class Conv2D(tf.keras.layers.Layer):
    """ Multiple repetitions of 2d convolution, batch normalization and dropout layers. """

    def __init__(self, filters, kernel_size=3, strides=1, padding='VALID', add_dropout=True, dropout_rate=0.2,
                 batch_normalization=False, num_repetitions=1):
        super().__init__()

        repetitions = []

        for i in range(num_repetitions):
            layer = []
            layer.append(tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                activation='relu'
            ))

            if batch_normalization:
                layer.append(tf.keras.layers.BatchNormalization())

            if add_dropout:
                layer.append(tf.keras.layers.Dropout(rate=dropout_rate))

            layer = tf.keras.Sequential(layer)

            repetitions.append(layer)

        self.combined_layer = tf.keras.Sequential(repetitions)

    def call(self, inputs, training=False):
        return self.combined_layer(inputs, training=training)


class Conv3D(tf.keras.layers.Layer):
    """ Multiple repetitions of 3d convolution, batch normalization and dropout layers. """

    def __init__(self, filters, kernel_size=3, strides=1, padding='VALID', add_dropout=True, dropout_rate=0.2,
                 batch_normalization=False, num_repetitions=1, convolve_time=True):
        super().__init__()

        repetitions = []

        t_size = kernel_size if convolve_time else 1
        kernel_shape = (t_size, kernel_size, kernel_size)

        for i in range(num_repetitions):
            layer = []
            layer.append(tf.keras.layers.Conv3D(
                filters=filters,
                kernel_size=kernel_shape,
                strides=strides,
                padding=padding,
                activation='relu'
            ))

            if batch_normalization:
                layer.append(tf.keras.layers.BatchNormalization())

            if add_dropout:
                layer.append(tf.keras.layers.Dropout(rate=dropout_rate))

            layer = tf.keras.Sequential(layer)

            repetitions.append(layer)

        self.combined_layer = tf.keras.Sequential(repetitions)

    def call(self, inputs, training=False):
        return self.combined_layer(inputs, training=training)


class Deconv2D(tf.keras.layers.Layer):
    """ 2d transpose convolution with optional batch normalization. """

    def __init__(self, filters, kernel_size=2, batch_normalization=False):
        super().__init__()

        layer = []
        layer.append(tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=kernel_size,
            padding='SAME',
            activation='relu'
        ))

        if batch_normalization:
            layer.append(tf.keras.layers.BatchNormalization())

        self.layer = tf.keras.Sequential(layer)

    def call(self, inputs, training=None):
        return self.layer(inputs, training=training)


class CropAndConcat(tf.keras.layers.Layer):
    """ Layer that crops the first tensor and concatenates it with the second. Used for skip connections. """
    @staticmethod
    def call(x1, x2):
        # Crop x1 to shape of x2
        x2_shape = tf.shape(x2)
        x1_crop = tf.image.resize_with_crop_or_pad(x1, x2_shape[1], x2_shape[2])

        # Concatenate along last dimension and return
        return tf.concat([x1_crop, x2], axis=-1)


class MaxPool3D(tf.keras.layers.Layer):
    def __init__(self, kernel_size=2, strides=2, pool_time=False):
        super().__init__()

        tsize = kernel_size if pool_time else 1
        tstride = strides if pool_time else 1

        kernel_shape = (tsize, kernel_size, kernel_size)
        strides = (tstride, strides, strides)

        self.layer = tf.keras.layers.MaxPool3D(
            pool_size=kernel_shape,
            strides=strides,
            padding='SAME'
        )

    def call(self, inputs, training=None):
        return self.layer(inputs, training=training)


class Reduce3DTo2D(tf.keras.layers.Layer):
    """ Reduces 3d representations into 2d using 3d convolution over the whole time dimension. """

    def __init__(self, filters, kernel_size=3, stride=1, add_dropout=False, dropout_rate=0.2):
        super().__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.add_dropout = add_dropout
        self.dropout_rate = dropout_rate
        self.layer = None

    def build(self, input_size):
        t_size = input_size[1]
        layer = []
        layer.append(tf.keras.layers.Conv3D(
            self.filters,
            kernel_size=(t_size, self.kernel_size, self.kernel_size),
            strides=(1, self.stride, self.stride),
            padding='VALID',
            activation='relu'
        ))

        if self.add_dropout:
            layer.append(tf.keras.layers.Dropout(rate=self.dropout_rate))

        self.layer = tf.keras.Sequential(layer)

    def call(self, inputs, training=None):
        r = self.layer(inputs, training=training)

        # Squeeze along temporal dimension
        return tf.squeeze(r, axis=[1])


# OLD, NOT YET CONVERTED

def conv2d_gru(input_, nfeats_out, k_size=3, scope='reduce_t', padding='VALID', return_sequence=False):
    """ Convolution GRU layer that takes as input a 5D tensor of shape [N, T, H, W, C] and operates over the temporal
        dimension. The resulting tensor has shape [N, H, W, C] if `return_sequence` is `False`, [N, T, H, W, C]
        otherwise.

        Cropping of the output tensor is applied if valid padding is chosen

        :param input_: Input 5D tensor of shape [N, T, H, W, C]
        :param nfeats_out: Number of channels/features in output tensor
        :param k_size: Dimension of filter kernel. Default is `3`
        :param scope: Scope of operation
        :param padding: Padding for output tensor. If `VALID`, the tensor is cropped along height and width
        :param return_sequence: Whether to return a 5D or 4D array (only last temporal frame considered)
        :return: Resulting tensor

    """
    with tf.variable_scope(scope):

        input_shape = input_.get_shape().as_list()

        cell = ConvGRUCell([input_shape[2], input_shape[3]], nfeats_out, [k_size, k_size])

        outputs, _ = tf.nn.dynamic_rnn(cell, input_, dtype=input_.dtype)

        if padding.lower() == 'valid':
            offs = (k_size-1)//2
            outputs = outputs[:, :, offs:-offs, offs:-offs, :]

        if return_sequence:
            return outputs

        return outputs[:, -1, :, :, :]


# This code is taken from the TF tutorial on transformers
# https://www.tensorflow.org/tutorials/text/transformer
def scaled_dot_product_attention(q, k, v, mask=None):
    """ Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training=None, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate=0.1, layer_norm=False):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.lnorm_in = tf.keras.layers.LayerNormalization() if layer_norm else None

        # replace embedding with 1d convolution
        self.conv1d = Conv1D(d_model, 1)
        # self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

        self.lnorm_out = tf.keras.layers.LayerNormalization() if layer_norm else None

    def call(self, x, training=None, mask=None):
        seq_len = tf.shape(x)[1]

        if self.lnorm_in:
            x = self.lnorm_in(x)

        # adding embedding and position encoding.
        x = self.conv1d(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        if self.lnorm_out:
            x = self.lnorm_out(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)
