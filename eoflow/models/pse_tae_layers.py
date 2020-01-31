import tensorflow as tf
import tensorflow.keras.layers as L

from .transformer_encoder_layers import scaled_dot_product_attention

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

    def call(self, x):
        return self.encoder(x)

class PseTaeMHAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = tf.reduce_mean(self.wq(q), axis=1, keepdims=True)  # (batch_size, 1, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, 1, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth_v)

        # scaled_attention.shape == (batch_size, num_heads, 1, depth_v)
        # attention_weights.shape == (batch_size, num_heads, 1, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # (batch_size, 1, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, 1, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # (batch_size, 1, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights

class TemporalAttentionEncoder(tf.keras.layers.Layer):
    def __init__(self, n_head=4, d_k=32, d_model=None, n_neurons=[512, 128, 128], dropout=0.2,
                 T=1000, len_max_seq=24):
        super().__init__()

        encoder = transformer_encoder_layers.Encoder(
            num_layers=1,
            d_model=d_model,
            num_heads=n_head,
            dff=d_k,
            maximum_position_encoding=len_max_seq,
            layer_norm=True,
            rate=dropout,
            T=T,
            multi_head_attention=PseTaeMHAttention)

        mlp_layers = [LinearLayer(out_dim) for out_dim in n_neurons]

        self.encoder = tf.keras.Sequential([encoder, *mlp_layers])

    def call(self, x):
        return self.encoder(x)


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

    def call(self, x):
        return self.pooling_method(x, axis=1)

class SummaryConcatenate(tf.keras.layers.Layer):
    """ Runs multiple summary layers on a single input and concatenates them. """
    def __init__(self, layers, axis=-1):
        super().__init__()

        self.layers = layers
        self.axis = axis

    def call(self, x):
        layer_outputs = [layer(x) for layer in self.layers]
        return L.concatenate(layer_outputs, axis=self.axis)
