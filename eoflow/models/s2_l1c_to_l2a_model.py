from .layers import conv1d, conv2d, deconv2d, crop_and_concat, max_pool_2d
from eoflow.base.base_model import BaseModel
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class S2L1CToL2AModel(BaseModel):
    """ Implementation of a vanilla Fully-Convolutional-Network (aka U-net) """
    def __init__(self, config):
        super(S2L1CToL2AModel, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_model(self):
        logging.debug("Building model")
        # placeholder for training flag
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        # placeholder for dropout
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # placeholder for input tensor
        self.x = tf.placeholder(tf.float32,
                                shape=[None, self.config.im_size[0], self.config.im_size[1], self.config.im_size[2]],
                                name='images')
        # placeholder for labels tensor
        self.y = tf.placeholder(tf.float32,
                                shape=[None, self.config.lb_size[0], self.config.lb_size[1], self.config.lb_size[2]],
                                name='labels')
        # define size of convolutional kernel and stride
        conv_size = self.config.conv_size
        conv_stride = self.config.conv_stride
        # define layers of FCN
        n_layers = self.config.n_layers
        # define size of deconvolution (should match size of pooling stride)
        deconv_size = self.config.deconv_size
        # define size of pooling kernel and stride
        pool_size = self.config.pool_size
        pool_stride = self.config.pool_stride
        # Encoding path
        # the number of features of the convolutional kernels is proportional to the square of the level
        # for instance, starting with 32 features at the first level (layer=0), there will be 64 features at layer=1 and
        # 128 features at layer=2
        connection_outputs = []
        for layer in range(n_layers):
            # compute number of features as a function of network depth level
            features = 2 ** layer * self.config.features_root
            if layer == 0:
                # if top layer, previous tensor is input data
                prev = self.x
            else:
                # if not top layer, previous tensor is pooling
                prev = pool
            # bank of two convolutional filters
            conv = conv2d(prev,
                          features,
                          is_training=self.is_training,
                          k_size=conv_size,
                          im_stride=conv_stride,
                          scope='encoding_' + str(layer),
                          add_dropout=self.config.add_dropout,
                          keep_prob=self.keep_prob,
                          add_bn=self.config.add_batch_norm,
                          bias_init=self.config.bias_init)
            connection_outputs.append(conv)
            # max pooling operation
            pool = max_pool_2d(conv,
                               ksize=pool_size,
                               stride=pool_stride)
        # bank of 2 convolutional filters at bottom of U-net.
        bottom = conv2d(pool,
                        2 ** n_layers * self.config.features_root,
                        is_training=self.config.is_training,
                        k_size=conv_size,
                        im_stride=conv_stride,
                        scope='bottom',
                        add_dropout=self.config.add_dropout,
                        keep_prob=self.keep_prob,
                        add_bn=self.config.add_batch_norm)
        # Decoding path
        # the decoding path mirrors the encoding path in terms of number of features per convolutional filter
        for layer in range(n_layers):
            # find corresponding level in decoding branch
            conterpart_layer = n_layers - 1 - layer
            # get same number of features as counterpart layer
            features = 2 ** conterpart_layer * self.config.features_root
            if layer == 0:
                prev = bottom
            else:
                prev = conv_decoding
            # transposed convolution to upsample tensors
            shape = prev.get_shape().as_list()
            deconv_output_shape = [tf.shape(prev)[0], shape[1] * deconv_size, shape[2] * deconv_size, features]
            deconv = deconv2d(prev,
                              deconv_output_shape,
                              k_size=deconv_size,
                              is_training=self.config.is_training,
                              scope='deconv_' + str(conterpart_layer),
                              add_bn=self.config.add_batch_norm)
            # skip connections to concatenate features from encoding path
            cc = crop_and_concat(connection_outputs[conterpart_layer],
                                 deconv)
            # bank of 2 convolutional filters
            conv_decoding = conv2d(cc,
                                   features,
                                   k_size=conv_size,
                                   im_stride=conv_stride,
                                   is_training=self.is_training,
                                   scope='decoding_' + str(conterpart_layer),
                                   add_dropout=self.config.add_dropout,
                                   keep_prob=self.keep_prob,
                                   add_bn=self.config.add_batch_norm)
        # final 1x1 convolution corresponding to pixel-wise linear combination of feature channels
        logits = conv1d(conv_decoding,
                        self.config.lb_size[2],
                        scope='logits',
                        bias_init=self.config.bias_init,
                        name='preds')
        # predicted bands are the output of the convolution
        self.preds = logits
        # compute classification accuracy
        self.accuracy = tf.constant(0.0, dtype=tf.float32)
        # loss is MAE or MSE (tf.losses.mean_squared_error)
        self.loss = tf.reduce_mean(tf.losses.absolute_difference(self.y, self.preds))
        # update operations for batch-normalisation and define train step as minimisation of loss
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss,
                                                                                         global_step=self.global_step_tensor)

    def init_saver(self):
        logging.debug("Initialising model saver")
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
