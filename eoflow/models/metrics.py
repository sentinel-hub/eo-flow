import tensorflow as tf


class InitializableMetric(tf.keras.metrics.Metric):
    """ Metric that has to be initialized from model configuration. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialized = False

    def init_from_config(self, model_config):
        """ Initializes the metric from configuration. """

        self.initialized = True

    def assert_initialized(self):
        """ Checks if the metric is initialized. """

        if not self.initialized:
            raise AssertionError("InitializableMetric was not initialized before use.")

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

    def init_from_config(self, model_config):
        super().init_from_config(model_config)

        if 'n_classes' in model_config:
            self.metric = tf.keras.metrics.MeanIoU(num_classes=model_config.n_classes)
        else:
            print("n_classes not found in model config. Using default max value.")
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
