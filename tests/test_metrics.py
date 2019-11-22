import unittest
import numpy as np

from eoflow.models.metrics import MeanIoU


class TestMeanIoU(unittest.TestCase):
    def test_not_initialized(self):
        metric = MeanIoU()

        y_true = np.zeros((1,32,32,3))
        y_pred = np.zeros((1,32,32,3))

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

        y_true1 = np.stack([ones, zeros], axis=-1) # All class 0
        y_true2 = np.stack([zeros, ones], axis=-1) # All class 1
        y_true3 = np.stack([mixed, 1-mixed], axis=-1) # Half class 1, half class 0

        # Check each one seperately
        metric.update_state(y_true1, y_pred)
        self.assertAlmostEqual(metric.result().numpy(), 0.0, 10)

        metric.reset_states()
        metric.update_state(y_true2, y_pred)
        self.assertAlmostEqual(metric.result().numpy(), 1.0, 10)

        metric.reset_states()
        metric.update_state(y_true3, y_pred)
        self.assertAlmostEqual(metric.result().numpy(), 0.25, 10) # Class 1 IoU: 0.5, Class 2 IoU: 0.0

        # Check aggregation
        metric.reset_states()
        metric.update_state(y_true1, y_pred)
        metric.update_state(y_true2, y_pred)
        metric.update_state(y_true3, y_pred)
        self.assertAlmostEqual(metric.result().numpy(), 0.25, 10) # Class 1 IoU: 0.5, Class 2 IoU: 0.0


if __name__ == '__main__':
    unittest.main()