import tensorflow as tf

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

