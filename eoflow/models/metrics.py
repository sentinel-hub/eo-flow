import warnings

from typing import Any, Callable, List

from skimage import measure
from scipy import ndimage

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


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

        self.metric = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=model_config['n_classes'])

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


class GeometricMetrics(InitializableMetric):
    """"
    Implementation of Geometric error metrics. Oversegmentation, Undersegmentation, Border, Fragmentation errors.

    The error metrics are based on a paper by C. Persello, A Novel Protocol for Accuracy Assessment in Classification of
    Very High Resolution Images (https://ieeexplore.ieee.org/document/5282610)
    """

    @staticmethod
    def _detect_edges(im: np.ndarray, thr: float = 0) -> np.ndarray:
        """ Edge detection function using the sobel operator. """
        sx = ndimage.sobel(im, axis=0, mode='constant')
        sy = ndimage.sobel(im, axis=1, mode='constant')
        sob = np.hypot(sx, sy)
        return sob > thr

    @staticmethod
    def _segmentation_error(intersection_area: float, object_area: float) -> float:
        return 1. - intersection_area / object_area

    @staticmethod
    def _intersection(mask1: np.ndarray, mask2: np.ndarray) -> float:
        return np.sum(np.logical_and(mask1, mask2))

    def _border_err(self, border_ref_edge: np.ndarray, border_meas_edge: np.ndarray) -> float:
        ref_edge_size = np.sum(border_ref_edge)
        intersection = self._intersection(border_ref_edge, border_meas_edge)
        err = intersection / ref_edge_size if ref_edge_size != 0 else 0
        be = 1. - err
        return be

    def _fragmentation_err(self, r: int, reference_mask: np.ndarray) -> float:
        if r <= 1:
            return 0
        den = np.sum(reference_mask) - self.pixel_size
        err = (r - 1.) / den if den > 0 else 0
        return err

    @staticmethod
    def _validate_input(reference, measurement):
        if np.ndim(reference) != np.ndim(measurement):
            raise ValueError("Reference and measurement input shapes must match.")

    def __init__(self, pixel_size: int = 1, edge_func: Callable = None, **edge_func_params: Any):

        super().__init__(name='geometric_metrics', dtype=tf.float64)

        self.oversegmentation_error = []
        self.undersegmentation_error = []
        self.border_error = []
        self.fragmentation_error = []

        self.edge_func = self._detect_edges if edge_func is None else edge_func
        self.edge_func_params = edge_func_params
        self.pixel_size = pixel_size

    def update_state(self, reference: np.ndarray, measurement: np.ndarray, encode_reference: bool = True,
                     background_value: int = 0) -> None:
        """ Calculate the error metrics for a measurement and reference arrays. For each .

        If encode_reference is set to True, connected components will be used to label objects in the reference and
        measurements.
        """

        if not tf.executing_eagerly():
            warnings.warn("Geometric metrics must be run with eager execution. If running as a compiled Keras model, "
                          "enable eager execution with model.run_eagerly = True")

        reference = reference.numpy() if isinstance(reference, tf.Tensor) else reference
        measurement = measurement.numpy() if isinstance(reference, tf.Tensor) else measurement

        self._validate_input(reference, measurement)

        for ref, meas in zip(reference, measurement):
            ref = ref
            meas = meas

            if encode_reference:
                cc_reference = measure.label(ref, background=background_value)
            else:
                cc_reference = ref

            cc_measurement = measure.label(meas, background=background_value)
            components_reference = set(np.unique(cc_reference)).difference([background_value])

            ref_edges = self.edge_func(cc_reference)
            meas_edges = self.edge_func(cc_measurement)
            for component in components_reference:
                reference_mask = cc_reference == component

                uniq, count = np.unique(cc_measurement[reference_mask & (cc_measurement != background_value)],
                                        return_counts=True)
                ref_area = np.sum(reference_mask)

                max_interecting_measurement = uniq[count.argmax()] if len(count) > 0 else background_value
                meas_mask = cc_measurement == max_interecting_measurement
                meas_area = np.count_nonzero(cc_measurement == max_interecting_measurement)
                intersection_area = count.max() if len(count) > 0 else 0

                self.oversegmentation_error.append(self._segmentation_error(intersection_area, ref_area))
                self.undersegmentation_error.append(self._segmentation_error(intersection_area, meas_area))
                border_ref_edge = ref_edges.squeeze() & reference_mask.squeeze()
                border_meas_edge = meas_edges.squeeze() & meas_mask.squeeze()

                self.border_error.append(self._border_err(border_ref_edge, border_meas_edge))
                self.fragmentation_error.append(self._fragmentation_err(len(uniq), reference_mask))

    def get_oversegmentation_error(self) -> float:
        """ Return oversegmentation error. """
        return np.array(self.oversegmentation_error).mean()

    def get_undersegmentation_error(self) -> float:
        """ Return undersegmentation error. """

        return np.array(self.undersegmentation_error).mean()

    def get_border_error(self) -> float:
        """ Return border error. """

        return np.array(self.border_error).mean()

    def get_fragmentation_error(self) -> float:
        """ Return fragmentation error. """

        return np.array(self.fragmentation_error).mean()

    def result(self) -> List[float]:
        """ Return a list  of values representing oversegmentation, undersegmentation, border, fragmentation errors. """

        return [self.get_oversegmentation_error(),
                self.get_undersegmentation_error(),
                self.get_border_error(), self.get_fragmentation_error()]

    def reset_states(self) -> None:
        """ Empty all the error arrays. """
        self.oversegmentation_error = []
        self.undersegmentation_error = []
        self.border_error = []
        self.fragmentation_error = []
