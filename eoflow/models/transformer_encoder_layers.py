import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Conv1D, LayerNormalization

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


def positional_encoding(positions, d_model, T=10000):

    if isinstance(positions, int):
        positions = np.arange(positions)
    else:
        positions = np.array(positions)

    def _get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(T, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    depths = np.arange(d_model)

    angle_rads = _get_angles(positions[:, np.newaxis],
                            depths[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, return_attention_masks, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.return_attention_masks = return_attention_masks
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=None, mask=None):
        attn_output, att_weights = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        if self.return_attention_masks:
            return out2, att_weights
        else:
            return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_shape, return_attention_masks=False, dropout_rate=0.1, layer_norm=False, T=10000, custom_pos_enc=False):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.return_attention_masks = return_attention_masks
        self.timeseries_length = input_shape[0]

        self.lnorm_in = tf.keras.layers.LayerNormalization() if layer_norm else None
        self.lnorm_conv = tf.keras.layers.LayerNormalization() if layer_norm else None
        
        self.custom_pos_enc = custom_pos_enc

        # replace embedding with 1d convolution
        self.conv_in = Conv1D(d_model, 1)
        enc_input_shape = (input_shape[0], d_model) # need output shape of this layer for encoder input

        if not self.custom_pos_enc:
            self.pos_encoding = positional_encoding(input_shape[0], self.d_model, T=T)

        encoder_layers = [EncoderLayer(d_model, num_heads, dff, return_attention_masks, dropout_rate)
                          for _ in range(num_layers)]
        if not self.return_attention_masks:
            self.encoder = tf.keras.Sequential(encoder_layers)
        else:
            enc_input = tf.keras.Input(shape=enc_input_shape)
            prev_tensor = enc_input
            att_outputs = []
            for el in encoder_layers:
                tmp = el(prev_tensor)
                att_outputs.extend(tmp[1:])
                prev_tensor = tmp[0]
            self.encoder = tf.keras.Model(inputs=enc_input, outputs=[prev_tensor] + att_outputs)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)


    def call(self, x, training=None, mask=None):
        seq_len = tf.shape(x)[1]

        # sort out x if custom pos encoding
        if self.custom_pos_enc:
            p = x[...,0]
            x = x[...,1:]
        else:
            p = self.timeseries_length

        if self.lnorm_in:
            x = self.lnorm_in(x)

        # adding embedding and position encoding.
        x = self.conv_in(x, training=training)  # (batch_size, input_seq_len, d_model)
        if self.lnorm_conv:
            x = self.lnorm_conv(x)

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        if self.custom_pos_enc:
            x += tf_positional_encoding(p, self.d_model, T=10000)[:, :seq_len, :]
        else:
            x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        x = self.encoder(x, training=training, mask=mask)

        return x  # (batch_size, input_seq_len, d_model)


def tf_positional_encoding(positions, d_model, T=10000):
    '''
    tensorflow implementation of positional encoding so works on input tensors

    positions will be samples x timesteps in shape
    the output pos enc will be samples x timesteps x d_model

    '''
    pos = tf.expand_dims(positions, axis=2)

    min_freq=1e-4 
    mask = tf.range(d_model)
    sin_mask = tf.cast(mask%2, tf.float32)
    cos_mask = 1-sin_mask
    exponent = 2*(mask//2)
    exponent = tf.cast(exponent, tf.float32)/tf.cast(d_model, tf.float32)
    freqs = min_freq**exponent

    angles = tf.einsum('ij,k->ijk', positions, freqs)
    pos_enc = tf.math.cos(angles)*cos_mask + tf.math.sin(angles)*sin_mask

    return pos_enc


class InputDropout(tf.keras.layers.Layer):
    """
    Drops out entire observations from the input (i.e. all bands for a single date)
    For each sample, random rate between min and max is taken, and obs are dropped out at that rate
    """
    def __init__(self, min_rate=0, max_rate=0.5):
        super(InputDropout, self).__init__()
        self.minrate = min_rate
        self.maxrate = max_rate
      
        
    def call(self, inputs, training=True):
        if training:
            #return tf.matmul(inputs, self.w) + self.b
            xdtype = inputs.dtype
            # create a rate variable with upper and lower bounds
            rate = tf.random.uniform((tf.shape(inputs)[0:1]), self.minrate, self.maxrate, dtype=xdtype)
            rate = tf.expand_dims(rate, -1)
            # create the random tensor
            random_tensor = tf.random.uniform(shape=(tf.shape(inputs)[0:2]), dtype=xdtype)

            keep_mask = random_tensor >= rate

            #ret = gen_math_ops.mul(ret, gen_math_ops.cast(keep_mask, x_dtype))
            ret = inputs * tf.cast(tf.expand_dims(keep_mask, -1), xdtype)

        else:
            ret = inputs

        return ret