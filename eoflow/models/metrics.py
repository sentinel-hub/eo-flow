import tensorflow as tf
import tensorflow_addons as tfa


class InitializableMetric(tf.keras.metrics.Metric):
    """ Metric that has to be initialized from model configuration. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialized = False

    def init_from_config(self, model_config=None):
        """ Initializes the metric from configuration. """

        self.initialized = True

    def assert_initialized(self):
        """ Checks if the metric is initialized. """

        if not self.initialized:
            raise ValueError("InitializableMetric was not initialized before use.")


class MeanIoU(InitializableMetric):
    """ Computes mean intersection over union metric for semantic segmentation.
    Wraps keras MeanIoU to work on logits. """

    def __init__(self, default_max_classes=32, name='mean_iou'):
        """ Creates MeanIoU metric

        :param default_max_classes: Default value for max number of classes. Required by Keras MeanIoU.
                                    Must be greater or equal to the actual number of classes.
                                    Will not be used if n_classes is in model configuration. Defaults to 32.
        :type default_max_classes: int
        :param name: Name of the metric
        :type name: str
        """

        super().__init__(name=name, dtype=tf.float32)
        self.default_max_classes = default_max_classes
        self.metric = None

    def init_from_config(self, model_config=None):
        super().init_from_config(model_config)

        if model_config is not None and 'n_classes' in model_config:
            self.metric = tf.keras.metrics.MeanIoU(num_classes=model_config['n_classes'])
        else:
            print("n_classes not found in model config or model config not provided. Using default max value.")
            self.metric = tf.keras.metrics.MeanIoU(num_classes=self.default_max_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.assert_initialized()

        y_pred_c = tf.argmax(y_pred, axis=-1)
        y_true_c = tf.argmax(y_true, axis=-1)

        return self.metric.update_state(y_true_c, y_pred_c, sample_weight)

    def result(self):
        self.assert_initialized()

        return self.metric.result()

    def reset_states(self):
        self.assert_initialized()

        return self.metric.reset_states()

    def get_config(self):
        self.assert_initialized()

        return self.metric.get_config()


class CroppedMetric(tf.keras.metrics.Metric):
    """ Wraps a metric. Crops the labels to match the logits size. """

    def __init__(self, metric):
        super().__init__(name=metric.name, dtype=metric.dtype)
        self.metric = metric

    def update_state(self, y_true, y_pred, sample_weight=None):
        logits_shape = tf.shape(y_pred)
        labels_crop = tf.image.resize_with_crop_or_pad(y_true, logits_shape[1], logits_shape[2])

        return self.metric.update_state(labels_crop, y_pred, sample_weight)

    def result(self):
        return self.metric.result()

    def reset_states(self):
        return self.metric.reset_states()

    def get_config(self):
        return self.metric.get_config()


class MCCMetric(InitializableMetric):
    """ Computes Mathew Correlation Coefficient metric. Wraps metrics.MatthewsCorrelationCoefficient from
    tensorflow-addons, and reshapes the input (logits) into (m, n_classes) tensors. The logits are thresholded to get
    "one-hot encoded" values for (multi)class metrics """

    def __init__(self, default_n_classes=2, default_threshold=0.5, name='mcc'):
        """ Creates MCCMetric metric

        :param default_n_classes: Default number of classes
        :type default_n_classes: int
        :param default_threshold: Default value for threshold
        :type default_threshold: float
        :param name: Name of the metric
        :type name: str
        """

        super().__init__(name=name, dtype=tf.float32)
        self.metric = None
        self.default_n_classes = default_n_classes
        self.threshold = default_threshold

    def init_from_config(self, model_config=None):
        super().init_from_config(model_config)

        if model_config is not None and 'n_classes' in model_config:
            self.metric = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=model_config['n_classes'])
        else:
            print("n_classes not found in model config or model config not provided. Using default max value.")
            self.metric = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=self.default_n_classes)

        if model_config is not None and 'mcc_threshold' in model_config:
            self.threshold = model_config['mcc_threshold']
        else:
            print(f"Using default value for threshold: {self.threshold}.")

        self.metric = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=self.default_n_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.assert_initialized()

        n = tf.math.reduce_prod(tf.shape(y_pred)[:-1])
        y_pred_c = tf.reshape(y_pred > self.threshold, (n, self.metric.num_classes))
        y_true_c = tf.reshape(y_true, (n, self.metric.num_classes))

        return self.metric.update_state(y_true_c, y_pred_c, sample_weight=sample_weight)

    def result(self):
        self.assert_initialized()

        return self.metric.result()

    def reset_states(self):
        self.assert_initialized()

        return self.metric.reset_states()

    def get_config(self):
        self.assert_initialized()

        return self.metric.get_config()
