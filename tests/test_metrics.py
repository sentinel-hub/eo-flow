import unittest
import numpy as np

import tensorflow as tf

from eoflow.models.metrics import MeanIoU, MCCMetric


class TestMeanIoU(unittest.TestCase):
    def test_not_initialized(self):
        metric = MeanIoU()

        y_true = np.zeros((1, 32, 32, 3))
        y_pred = np.zeros((1, 32, 32, 3))

        # Errors should be raised (because not initialized)
        self.assertRaises(ValueError, metric.update_state, y_true, y_pred)
        self.assertRaises(ValueError, metric.result)
        self.assertRaises(ValueError, metric.reset_states)
        self.assertRaises(ValueError, metric.get_config)

        metric.init_from_config()

        # Test that errors are not raised
        metric.update_state(y_true, y_pred)
        metric.result()
        metric.reset_states()
        metric.get_config()

    def test_iou_results(self):
        metric = MeanIoU()
        metric.init_from_config({'n_classes': 3})

        ones = np.ones((32, 32))
        zeros = np.zeros((32, 32))
        mixed = np.concatenate([ones[:16], zeros[:16]])

        # Predict everything as class 1
        y_pred = np.stack([zeros, ones], axis=-1)

        y_true1 = np.stack([ones, zeros], axis=-1)  # All class 0
        y_true2 = np.stack([zeros, ones], axis=-1)  # All class 1
        y_true3 = np.stack([mixed, 1 - mixed], axis=-1)  # Half class 1, half class 0

        # Check each one seperately
        metric.update_state(y_true1, y_pred)
        self.assertAlmostEqual(metric.result().numpy(), 0.0, 10)

        metric.reset_states()
        metric.update_state(y_true2, y_pred)
        self.assertAlmostEqual(metric.result().numpy(), 1.0, 10)

        metric.reset_states()
        metric.update_state(y_true3, y_pred)
        self.assertAlmostEqual(metric.result().numpy(), 0.25, 10)  # Class 1 IoU: 0.5, Class 2 IoU: 0.0

        # Check aggregation
        metric.reset_states()
        metric.update_state(y_true1, y_pred)
        metric.update_state(y_true2, y_pred)
        metric.update_state(y_true3, y_pred)
        self.assertAlmostEqual(metric.result().numpy(), 0.25, 10)  # Class 1 IoU: 0.5, Class 2 IoU: 0.0


class TestMCC(unittest.TestCase):
    def test_not_initialized(self):
        metric = MCCMetric()

        y_true = np.zeros((1, 32, 32, 3))
        y_pred = np.zeros((1, 32, 32, 3))

        # Errors should be raised (because not initialized)
        self.assertRaises(ValueError, metric.update_state, y_true, y_pred)
        self.assertRaises(ValueError, metric.result)
        self.assertRaises(ValueError, metric.reset_states)
        self.assertRaises(ValueError, metric.get_config)

        metric.init_from_config({'n_classes': 3})

        # Test that errors are not raised
        metric.update_state(y_true, y_pred)
        metric.result()
        metric.reset_states()
        metric.get_config()

    def test_wrong_n_classes(self):
        metric = MCCMetric()

        n_classes = 3
        y_true = np.zeros((1, 32, 32, n_classes))
        y_pred = np.zeros((1, 32, 32, n_classes))

        metric.init_from_config({'n_classes': 1})

        # Test that errors are raised
        with self.assertRaises(Exception) as context:
            metric.update_state(y_true, y_pred)
            self.assertTrue((f'Input to reshape is a tensor with {np.prod(y_true.shape)} values, '
                             f'but the requested shape has {np.prod(y_true.shape[:-1])}') in str(context.exception))

    def test_mcc_results_binary_symmetry(self):
        metric = MCCMetric()
        metric.init_from_config({'n_classes': 2})

        y_pred = np.random.randint(0, 2, (32, 32, 1))
        y_pred = np.concatenate((y_pred, 1-y_pred), axis=-1)

        y_true = np.random.randint(0, 2, (32, 32, 1))
        y_true = np.concatenate((y_true, 1 - y_true), axis=-1)

        metric.update_state(y_true, y_pred)
        results = metric.result().numpy()
        self.assertAlmostEqual(results[0], results[1], 7)

    def test_mcc_single_vs_binary(self):
        metric_single = MCCMetric()
        metric_single.init_from_config({'n_classes': 1})

        y_pred = np.random.randint(0, 2, (32, 32, 1))
        y_true = np.random.randint(0, 2, (32, 32, 1))
        metric_single.update_state(y_true, y_pred)
        result_single = metric_single.result().numpy()[0]

        metric_binary = MCCMetric()
        metric_binary.init_from_config({'n_classes': 2})

        y_pred = np.concatenate((y_pred, 1-y_pred), axis=-1)
        y_true = np.concatenate((y_true, 1 - y_true), axis=-1)
        metric_binary.update_state(y_true, y_pred)
        result_binary = metric_binary.result().numpy()[0]

        self.assertAlmostEqual(result_single, result_binary, 7)

    def test_mcc_results(self):
        # test is from an example of MCC in sklearn.metrics matthews_corrcoef
        y_true = np.array([1, 1, 1, 0])[..., np.newaxis]
        y_pred = np.array([1, 0, 1, 1])[..., np.newaxis]
        metric = MCCMetric()
        metric.init_from_config({'n_classes': 1})
        metric.update_state(y_true, y_pred)
        self.assertAlmostEqual(metric.result().numpy()[0], -0.3333333, 7)

    def test_mcc_threshold(self):
        y_true = np.array([1, 1, 1, 0])[..., np.newaxis]
        y_pred = np.array([0.9, 0.6, 0.61, 0.7])[..., np.newaxis]
        metric = MCCMetric()
        metric.init_from_config({'n_classes': 1, 'mcc_threshold': 0.6})
        metric.update_state(y_true, y_pred)
        self.assertAlmostEqual(metric.result().numpy()[0], -0.3333333, 7)

    def test_fit(self):
        input_shape = (1, 28, 28, 2)

        mcc = MCCMetric()
        mcc.init_from_config({'n_classes': 2})

        iou = MeanIoU()
        iou.init_from_config({'n_classes': 2})

        model = tf.keras.Sequential([
            tf.keras.layers.Softmax()
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=[mcc, iou])

        model.fit(np.random.random(input_shape), np.ones(input_shape), epochs=1)


if __name__ == '__main__':
    unittest.main()
