import tensorflow as tf
from tensorflow.keras.losses import Loss, Reduction


class CategoricalFocalLoss(Loss):
    """ Categorical version of focal loss.

    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        Keras implementation: https://github.com/umbertogriffo/focal-loss-keras
    """

    def __init__(self, gamma=2., alpha=.25, from_logits=True, reduction=Reduction.AUTO, name='FocalLoss'):
        """Categorical version of focal loss.

        :param gamma: gamma value, defaults to 2.
        :type gamma: float
        :param alpha: alpha value, defaults to .25
        :type alpha: float
        :param from_logits: Whether predictions are logits or softmax, defaults to True
        :type from_logits: bool
        :param reduction: reduction to be used, defaults to Reduction.AUTO
        :type reduction: tf.keras.losses.Reduction, optional
        :param name: name of the loss, defaults to 'FocalLoss'
        :type name: str
        """
        super().__init__(reduction=reduction, name=name)

        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):

        # Perform softmax
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.)

        # Calculate Cross Entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate Focal Loss
        loss = self.alpha * tf.math.pow(1 - y_pred, self.gamma) * cross_entropy

        # Sum over classes
        loss = tf.reduce_sum(loss, axis=-1)

        return loss


class JaccardDistanceLoss(Loss):
    """ Implementation of the Jaccard distance, or Intersection over Union IoU loss.

    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    Implementation taken from https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
    """
    def __init__(self, smooth=1, from_logits=True, reduction=Reduction.AUTO, name='JaccardLoss'):
        """ Jaccard distance loss.

        :param smooth: Smoothing factor. Default is 1.
        :type smooth: int
        :param from_logits: Whether predictions are logits or softmax, defaults to True
        :type from_logits: bool
        :param reduction: reduction to be used, defaults to Reduction.AUTO
        :type reduction: tf.keras.losses.Reduction, optional
        :param name: name of the loss, defaults to 'JaccardLoss'
        :type name: str
        """
        super().__init__(reduction=reduction, name=name)

        self.smooth = smooth
        self.from_logits = from_logits

    def call(self, y_true, y_pred):

        # Perform softmax
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)

        intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2))

        sum_ = tf.reduce_sum(y_true + y_pred, axis=(1, 2))

        jac = (intersection + self.smooth) / (sum_ - intersection + self.smooth)

        loss = tf.reduce_mean((1 - jac) * self.smooth)

        return loss
