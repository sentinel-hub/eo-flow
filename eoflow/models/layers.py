from .conv_cells import ConvGRUCell
import tensorflow as tf
import numpy as np


def _conv2d(tin, nfeats_out, is_training, k_size, stride, varnames, add_bn=True, bias_init=0.0, padding='VALID'):
    """ Base 2D convolutional layer

        :param tin: Input TF tensor of shape [minibatch,h,w,nfeats]
        :type tin: 4D TF tensor
        :param nfeats_out: Number of features for convolutional layer
        :type nfeats_out: int
        :param is_training: Whether is training or testing for batch normalisation
        :type is_training: bool
        :param k_size: Size of convolutional kernel
        :type k_size: int
        :param stride: Stride of convolutional kernel
        :type stride: int
        :param varnames: Name of variable for weight, bias and batch normalisation tensors
        :type varnames: dict
        :param bias_init: Initialisation value for bias parameters. Default is `0.0`
        :type bias_init: float
        :param padding: Defines padding performed by convolution. Default is 'VALID'
        :type padding: str
        :return: Tensor resulting from convolution
        :rtype: 4D TF tensor
    """
    # Weights of convolutional layer
    w = tf.get_variable(varnames['w'],
                        [k_size, k_size, tin.get_shape()[-1], nfeats_out],
                        initializer=tf.keras.initializers.he_uniform())

    # Stride is [stride_minibatch, stride_height, stride_width, stride_channels]
    conv = tf.nn.conv2d(tin,
                        w,
                        strides=[1, stride, stride, 1],
                        padding=padding)

    # Bias of convolutional layer
    b = tf.get_variable(varnames['b'],
                        [nfeats_out],
                        initializer=tf.constant_initializer(bias_init))
    conv = tf.nn.bias_add(conv, b)

    # Batch normalisation
    if add_bn:
        conv = tf.layers.batch_norm(conv, training=is_training)

    # ReLU activation function
    r = tf.nn.relu(conv)
    return r


def _conv3d(tin, nfeats_out, is_training, k_size, stride, varnames, add_bn=False, convolve_time=True, bias_init=0.0,
            padding='VALID'):
    """ Base 3D convolutional layer

        :param tin: Input TF tensor of shape [minibatch, t, h, w, nfeats]
        :type tin: 5D TF tensor
        :param nfeats_out: Number of features to be computed
        :type nfeats_out: int
        :param is_training: Whether is training or testing for batch normalisation
        :type is_training: bool
        :param k_size: Size of convolutional layer. Actual kernel size will be [k_size, k_size, k_size]
        :type k_size: int
        :param stride: Stride of convolutional kernel
        :type stride: int
        :param varnames: Name of variable for weight, bias and batch normalisation tensors
        :type varnames: dict
        :param add_bn: Whether to add batch-normalisation. Default is `False`
        :type add_bn: bool
        :param bias_init: Initialisation value for bias parameters. Default is `0.0`
        :type bias_init: float
        :return: Tensor resulting from convolution
        :rtype: 5D TF tensor
    """
    t_size = k_size if convolve_time else 1

    # Weights of convolutional layer
    w = tf.get_variable(varnames['w'],
                        [t_size, k_size, k_size, tin.get_shape()[-1], nfeats_out],
                        initializer=tf.keras.initializers.he_uniform())

    # Stride is [stride_minibatch, stride_time, stride_height, stride_width, stride_channels]
    conv = tf.nn.conv3d(tin,
                        w,
                        strides=[1, 1, stride, stride, 1],
                        padding=padding)

    # Bias of convolutional layer
    b = tf.get_variable(varnames['b'],
                        [nfeats_out],
                        initializer=tf.constant_initializer(bias_init))
    conv = tf.nn.bias_add(conv, b)

    # Batch normalisation
    if add_bn:
        conv = tf.layers.batch_norm(conv, training=is_training)

    # ReLU activation function
    r = tf.nn.relu(conv)

    return r


def conv2d(input_, output_dim, is_training, k_size=3, im_stride=1, scope='conv2d', add_dropout=True, keep_prob=0.8,
           add_bn=False, single_filter=False, bias_init=0.0, padding='VALID'):
    """ Define a sequence of 2 2d convolutional layers

        :param input_: Input 4D tensor of shape NHWC
        :type input_: 4D TF tensor
        :param output_dim: Number of features/channels for the convolutional filter
        :type output_dim: int
        :param is_training: Boolean specifying training/testing
        :type is_training: bool
        :param k_size: Dimension of filter kernel. Default is `3`
        :type k_size: int
        :param im_stride: Stride of convolution along the feature image. Default is `1`
        :type im_stride: int
        :param scope: Defines scope of variables. Default is `'conv3d'`
        :type scope: str
        :param add_bn: Whether to add batch-normalisation or not. Defailt is `False`
        :type add_bn: bool
        :param add_dropout: Whether to add dropout to convolutional layers or not. Default is 'True'
        :type add_dropout: bool
        :param keep_prob: Ratio of neurons to keep in drop-out. Default is `0.8`
        :type keep_prob: float
        :param single_filter: Whether to apply 1 or 2 banks of conv filters. Default is `False`
        :type single_filter: bool
        :param bias_init: Initialisation value for bias parameters. Default is `0.0`
        :type bias_init: float
        :param padding: Defines padding performed by convolution. Default is 'VALID'
        :type padding: str
        :return r2: Result of convolutions

    """
    dropout_rate = 1 - keep_prob
    with tf.variable_scope(scope):
        # First convolutional filter
        c_2d_1 = {'w': 'w_2d_1', 'b': 'b_2d_1', 'bn': 'bn_2d_1'}
        r_2d_1 = _conv2d(input_,
                         output_dim,
                         is_training,
                         k_size,
                         im_stride,
                         c_2d_1,
                         add_bn=add_bn,
                         bias_init=bias_init,
                         padding=padding)
        if add_dropout:
            r_2d_1 = tf.layers.dropout(r_2d_1, rate=dropout_rate, training=is_training)

        if single_filter:

            return r_2d_1

        else:
            # Second convolutional filter
            c_2d_2 = {'w': 'w_2d_2', 'b': 'b_2d_2', 'bn': 'bn_2d_2'}
            r_2d_2 = _conv2d(r_2d_1,
                             output_dim,
                             is_training,
                             k_size,
                             im_stride,
                             c_2d_2,
                             add_bn=add_bn,
                             bias_init=bias_init,
                             padding=padding)
            if add_dropout:
                r_2d_2 = tf.layers.dropout(r_2d_2, rate=dropout_rate, training=is_training)

            return r_2d_2


def conv3d(input_, output_dim, is_training, k_size=3, im_stride=1, scope='conv3d', add_dropout=True, keep_prob=0.8,
           add_bn=False, single_filter=False, convolve_time=True, bias_init=0.0, padding='VALID'):
    """ Define a sequence of 1 or 2 3d convolutional layers to be used in encoding path

        :param input_: Input 5D tensor of shape [N x T x H x W x C]
        :type input_: 5D TF tensor
        :param output_dim: Number of features/channels for the convolutional filter
        :type output_dim: int
        :param is_training: Boolean specifying training/testing
        :type is_training: bool
        :param k_size: Dimension of filter kernel. Default is `3`
        :type k_size: int
        :param im_stride: Stride of convolution along the feature image. Temporal stride is fixed to 1. Default is `1`
        :type im_stride: int
        :param scope: Scope of variables. Default is `'conv3d'`
        :type scope: str
        :param add_dropout: Whether to add dropout to convolutional filter. Default is 'True'
        :type add_dropout: bool
        :param keep_prob: Ratio of neurons to keep if dropout is used. Default is `0.8`
        :type keep_prob: float
        :param add_bn: Whether to add batch-normalisation to the convolutional layers. Defailt is `False`
        :type add_bn: bool
        :param single_filter: Whether to apply 1 or 2 banks of conv filters. Default is `False`
        :type single_filter: bool
        :param convolve_time: Whether we want to apply convolution along the dime dimension (actual 3D convolution).
                                Default is `True`
        :param bias_init: Initialisation value for bias parameters. Default is `0.0`
        :type bias_init: float
        :param padding: Defines padding performed by convolution. Default is 'VALID'
        :type padding: str
        :return: Result of convolutions
    """
    dropout_rate = 1 - keep_prob

    with tf.variable_scope(scope):
        # First 3D convolution
        c_3d_1 = {'w': 'w_3d_1', 'b': 'b_3d_1', 'bn': 'bn_3d_1'}
        r_3d_1 = _conv3d(input_,
                         output_dim,
                         is_training,
                         k_size,
                         im_stride,
                         c_3d_1,
                         add_bn=add_bn,
                         convolve_time=convolve_time,
                         bias_init=bias_init,
                         padding=padding)
        if add_dropout:
            r_3d_1 = tf.layers.dropout(r_3d_1, rate=dropout_rate, training=is_training)

        if single_filter:

            return r_3d_1
        else:
            # Second convolutional layer
            c_3d_2 = {'w': 'w_3d_2', 'b': 'b_3d_2', 'bn': 'bn_3d_2'}
            r_3d_2 = _conv3d(r_3d_1,
                             output_dim,
                             is_training,
                             k_size,
                             im_stride,
                             c_3d_2,
                             add_bn=add_bn,
                             convolve_time=convolve_time,
                             bias_init=bias_init,
                             padding=padding)
            if add_dropout:
                r_3d_2 = tf.layers.dropout(r_3d_2, rate=dropout_rate, training=is_training)

            return r_3d_2


def deconv2d(input_, output_shape, is_training, k_size=2, scope='deconv2d', add_bn=True):
    """ Transposed convolution layer to perform upsampling

        :param add_bn:
        :param input_: Input tensor
        :param output_shape: Shape of output tensor
        :param is_training: Flag to tell batch normalisation if training
        :param k_size: Size of kernel to perform deconvolution (default=2)
        :param scope: Sets scope of the variables
        :return r: Output of deconvolution

    """
    with tf.variable_scope(scope):

        output_dim = output_shape[-1]

        # Weights tensor of deconvolutional layer
        w = tf.get_variable('w', [k_size, k_size, output_dim, input_.get_shape()[-1]],
                            initializer=tf.keras.initializers.he_uniform())

        # Deconvolutional layer
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape, strides=[1, k_size, k_size, 1], padding='SAME')

        # Batch-normalisation
        if add_bn:
            deconv = tf.layers.batch_norm(deconv, training=is_training)

        # ReLU activation function
        r = tf.nn.relu(deconv)

        return r


def crop_and_concat(x1, x2):
    """ Function to generate skip connections, cropping and concatenating tensors

        :param x1: First tensor to be cropped and concatenated
        :param x2: Second tensor to be concatenated
        :return out: Output tensor

    """
    # Get shapes of tensors
    x1_shape = x1.get_shape().as_list()
    x2_shape = x2.get_shape().as_list()

    # Get differences in shapes
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]

    # Crop the first tensor
    x1_crop = tf.slice(x1, offsets, size)

    # Concatenate along last dimension and return
    return tf.concat([x1_crop, x2], axis=-1)


def max_pool_2d(input_, ksize=2, stride=2):
    """ Max pooling of a 4D tensor which represents a 2D (multi-channel) input image

        :param input_: Input 4D tensor of shape [N, H, W, C]
        :param ksize: Size of pooling kernel along height and width of tensor. Default is `2`
        :param stride: Stride of pooling along height and width of tensor. Default is `2`
        :return: Tensor resulting form max pooling
    """
    return tf.nn.max_pool(input_,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')


def max_pool_3d(input_, ksize=2, stride=2, pool_time=False):
    """ Max pooling of a 5D tensor which represents either a 2D+t or 3D (multi-channel) input image

        :param input_: Input 5D tensor of shape [N, D, H, W, C]. D could be time or depth of image.
        :param ksize: Size of pooling kernel. If `pool_time` is `True`, the kernel size along D is `1`, otherwise is
                        equal to ksize. Default is `2`
        :param stride: Stride of pooling kernel. If `pool_time` is `True`, the kernel size along D is `1`, otherwise is
                        equal to stride. Default is `2`
        :param pool_time: Whether to operate pooling along the 2nd dimension (time or depth). Default is `False`
        :return: Tensor resulting from max pooling
    """
    ktsize = ksize if pool_time else 1
    tstride = stride if pool_time else 1

    return tf.nn.max_pool3d(input_,
                            ksize=[1, ktsize, ksize, ksize, 1],
                            strides=[1, tstride, stride, stride, 1],
                            padding='SAME')


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


def reduce_3d_to_2d(input_, nfeats_out, k_size=3, im_stride=1, add_dropout=False, keep_prob=.8, scope='reduce_t',
                    bias_init=0.0, padding='VALID', is_training=True):
    """ Reduce Spatio-temporal 3d volume to spatial 2d image by 3d convolution over time and axis squeezing

        This function reduces a 5D TF tensor of shape [N, T, H, W, C] to a 4D tensor of shape [N, H, W, C]

        :param input_: 5D TF tensor of shape [N, T, H, W, C]
        :type input_: TF tensor
        :param nfeats_out: Number of channels in output tensor
        :type nfeats_out: int
        :param k_size: Dimension of filter kernel. Default is `3`
        :type k_size: int
        :param im_stride: Stride of convolution along the feature image. Temporal stride is fixed to 1. Default is `1`
        :type im_stride: int
        :param add_dropout: Whether ot add dropout to convolutional layer
        :type add_dropout: bool
        :param keep_prob: Ratio of neurons to keep during dropout
        :type keep_prob: float
        :param scope: Scope of operation
        :type scope: str
        :param bias_init: Initialisation value for bias parameters. Default is `0.0`
        :type bias_init: float
        :param padding: Defines padding performed by convolution. Default is 'VALID'
        :type padding: str
        :return: 4D tensor of shape [N, H, W, C]
        :rtype: TF tensor
    """
    dropout_rate = 1 - keep_prob
    with tf.variable_scope(scope):
        # Shape of input tensor
        input_shape = input_.get_shape().as_list()

        # convolution along time
        w = tf.get_variable('w_3d_to_2d', [input_shape[1], k_size, k_size, input_shape[-1], nfeats_out],
                            initializer=tf.keras.initializers.he_uniform())
        # convolution
        conv = tf.nn.conv3d(input_, w, strides=[1, 1, im_stride, im_stride, 1], padding=padding)

        # Bias of convolutional layer
        b = tf.get_variable('b_3d_to_2d', [nfeats_out], initializer=tf.constant_initializer(bias_init))

        conv = tf.nn.bias_add(conv, b)

        # ReLU activation function
        r = tf.nn.relu(conv)

        # Dropout
        if add_dropout:
            r = tf.layers.dropout(r, rate=dropout_rate, training=is_training)

        # Squeeze along temporal dimension
        out = tf.squeeze(r, axis=[1], name='squeeze')

        return out


def conv1d(input_, nfeats_out, scope="logits", bias_init=0.0, name=None):
    """ Run final 1x1 convolution on a 4D tensor of shape [N, H, W, D]

        This convolution reduces the last dimension D to `nfeats_out` channel, typically corresponding to the number of
        classes
    """
    with tf.variable_scope(scope):
        w = tf.get_variable('w',
                            [1, 1, input_.get_shape()[-1], nfeats_out],
                            initializer=tf.keras.initializers.he_uniform())
        logits = tf.nn.conv2d(input_,
                              w,
                              strides=[1, 1, 1, 1],
                              padding='SAME')
        b = tf.get_variable('b',
                            [nfeats_out],
                            initializer=tf.constant_initializer(bias_init))
        logits = tf.nn.bias_add(logits, b, name=name)

        return logits


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
