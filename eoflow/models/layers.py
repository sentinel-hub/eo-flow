from .conv_cells import ConvGRUCell
import tensorflow as tf
import numpy as np


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


def weighted_cross_entropy(flat_logits, flat_labels, class_weights):
    """ Compute weighted cross-entropy assigning different weights to different classes

        Labels and digits need be flattened, i.e. 2-dimensional with second dimension equal to the number of classes
    """
    class_weights = tf.constant(np.asarray(class_weights, dtype=np.float32))
    weight_map = tf.reduce_max(tf.multiply(flat_labels, class_weights), axis=1)
    loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits, labels=flat_labels)
    weighted_loss = tf.multiply(loss_map, weight_map)
    loss = tf.reduce_mean(weighted_loss)
    return loss


def compute_iou_loss(n_classes, probs, preds, labels, class_weights=None, exclude_background=True):
    """ Compute Intersection-Over-Union (Jaccard metric) loss

    This loss is proportional to the Dice score, but is actually a metric
    """
    eps = 1e-5
    iou_loss = 0
    classes = np.arange(1, n_classes) if exclude_background else np.arange(n_classes)
    # valid_classes = 0
    for i in classes:  # Loop through classes excluding back-ground
        slice_prob = tf.squeeze(tf.slice(probs, [0, 0, 0, i], [-1, -1, -1, 1]), axis=-1)
        slice_pred = tf.cast(tf.equal(preds, i), tf.float32)
        slice_label = tf.squeeze(tf.slice(labels, [0, 0, 0, i], [-1, -1, -1, 1]), axis=-1)
        intersection_prob = tf.reduce_sum(tf.multiply(slice_prob, slice_label), axis=[1, 2])
        intersection_pred = tf.reduce_sum(tf.multiply(slice_pred, slice_label), axis=[1, 2])
        union = eps + tf.reduce_sum(slice_pred, axis=[1, 2]) + tf.reduce_sum(slice_label, axis=[1, 2]) \
                - intersection_pred
        # this is mean over batch
        multiplier = 1/len(classes) if class_weights is None else class_weights[i]/np.sum(class_weights)
        iou_loss += multiplier * tf.reduce_mean(tf.math.divide(intersection_prob, union))
        # # only labels appearing in the batch count towards the loss (e.g. tundra or wetland are not considered)
        # valid_classes = tf.cond(tf.reduce_sum(slice_label) > 0, lambda: tf.add(valid_classes, 1),
        #                         lambda: tf.add(valid_classes, 0))
    # # catch the case where there are no labels at all in ground-truth (just no-data)
    # valid_classes = tf.cond(tf.equal(valid_classes, 0), lambda: 1, lambda: valid_classes)
    iou_loss = 1 - iou_loss
    return iou_loss
