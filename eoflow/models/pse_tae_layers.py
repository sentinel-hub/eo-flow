import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L

from .transformer_encoder_layers import scaled_dot_product_attention, positional_encoding

pooling_methods = {
    'mean': tf.math.reduce_mean,
    'std': tf.math.reduce_std,
    'max': tf.math.reduce_max,
    'min': tf.math.reduce_min
}

class PixelSetEncoder(tf.keras.layers.Layer):
    def __init__(self, mlp1=[10, 32, 64], mlp2=[64, 128], pooling='mean_std'):
        super().__init__()

        self.mlp1 = tf.keras.Sequential([LinearLayer(out_dim) for out_dim in mlp1])

        pooling_methods = [SetPooling(method) for method in pooling.split('_')]
        self.pooling = SummaryConcatenate(pooling_methods, axis=-1)

        mlp2_layers = [LinearLayer(out_dim) for out_dim in mlp2[:-1]]
        mlp2_layers.append(LinearLayer(mlp2[-1], activation=False))
        self.mlp2 = tf.keras.Sequential(mlp2_layers)

        self.encoder = tf.keras.Sequential([
            self.mlp1,
            self.pooling,
            self.mlp2
        ])

    def call(self, x, training=None, mask=None):
        return self.encoder(x, training=training, mask=mask)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, n_head, d_k, name='multi_head_attention'):
        super().__init__(name=name)

        self.n_head = n_head
        self.d_k = d_k

        self.fc1_q = L.Dense(d_k * n_head,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2.0 / d_k)))

        self.fc1_k = L.Dense(d_k * n_head,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2.0 / d_k)))

        self.fc2 = tf.keras.Sequential([
            L.BatchNormalization(),
            L.Dense(d_k)
        ])

    def split_heads(self, x, batch_size):
        """Split the last dimension into (n_head, d_k).
        Transpose the result such that the shape is (batch_size, n_head, seq_len, d_k)
        """

        x = tf.reshape(x, (batch_size, -1, self.n_head, self.d_k))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, training=None, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.fc1_q(q)
        q = self.split_heads(q, batch_size)
        q = tf.reduce_mean(q, axis=2, keepdims=True) # MEAN query

        k = self.fc1_k(k)
        k = self.split_heads(k, batch_size)

        # Repeat n_head times
        v = tf.expand_dims(v, axis=1)
        v = tf.tile(v, (1, self.n_head, 1, 1))

        output, attn = scaled_dot_product_attention(q, k, v, mask)

        output = tf.squeeze(output, axis=2)

        # Concat heads
        output = tf.reshape(output, (batch_size, -1))

        return output

class TemporalAttentionEncoder(tf.keras.layers.Layer):
    def __init__(self, n_head=4, d_k=32, d_model=None, n_neurons=[512, 128, 128], dropout=0.2,
                 T=1000, len_max_seq=24, positions=None):
        super().__init__()


        self.positions = positions
        if self.positions is None:
            self.positions = len_max_seq + 1

        self.d_model = d_model
        self.T = T

        self.in_layer_norm = tf.keras.layers.LayerNormalization(name='in_layer_norm')

        self.inconv = None
        if d_model is not None:
            self.inconv = tf.keras.Sequential([
                L.Conv1D(d_model, 1, name='inconv'),
                L.LayerNormalization(name='conv_layer_norm')
            ])

        self.out_layer_norm = tf.keras.layers.LayerNormalization(name='out_layer_norm')

        self.attention_heads = MultiHeadAttention(n_head, d_k, name='attention_heads')

        mlp_layers = [LinearLayer(out_dim) for out_dim in n_neurons]
        self.mlp = tf.keras.Sequential(mlp_layers, name='mlp')

        self.dropout = L.Dropout(dropout)

    def build(self, input_shape):
        d_in = input_shape[-1] if self.d_model is None else self.d_model
        self.position_enc = positional_encoding(self.positions, d_in, T=self.T)

    def call(self, x, training=None, mask=None):
        seq_len = tf.shape(x)[1]

        x = self.in_layer_norm(x, training=training)

        if self.inconv is not None:
            x = self.inconv(x, training=training)

        pos_encoding = self.position_enc[:, :seq_len, :]
        if self.positions is None:
            pos_encoding = self.position_enc[:, 1:seq_len+1, :]

        enc_output = x + pos_encoding

        enc_output = self.attention_heads(enc_output, enc_output, enc_output, training=training, mask=mask)

        enc_output = self.mlp(enc_output, training=training)
        enc_output = self.dropout(enc_output, training=training)
        enc_output = self.out_layer_norm(enc_output, training=training)

        return enc_output

def LinearLayer(out_dim, batch_norm=True, activation=True):
    """ Linear layer. """

    layers = [L.Dense(out_dim)]

    if batch_norm:
        layers.append(L.BatchNormalization())

    if activation:
        layers.append(L.ReLU())

    return tf.keras.Sequential(layers)

class SetPooling(tf.keras.layers.Layer):
    """ Pooling over the Set dimension using a specified pooling method. """
    def __init__(self, pooling_method):
        super().__init__()

        self.pooling_method = pooling_methods[pooling_method]

    def call(self, x, training=None, mask=None):
        return self.pooling_method(x, axis=1)

class SummaryConcatenate(tf.keras.layers.Layer):
    """ Runs multiple summary layers on a single input and concatenates them. """
    def __init__(self, layers, axis=-1):
        super().__init__()

        self.layers = layers
        self.axis = axis

    def call(self, x, training=None, mask=None):
        layer_outputs = [layer(x, training=training, mask=mask) for layer in self.layers]
        return L.concatenate(layer_outputs, axis=self.axis)
