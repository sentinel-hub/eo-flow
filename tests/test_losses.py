import unittest
import numpy as np

from eoflow.models.losses import CategoricalFocalLoss


class TestFocalLoss(unittest.TestCase):
    def test_shapes(self):
        loss_fn = CategoricalFocalLoss(from_logits=True)

        ones_1 = np.ones((1, 1024, 2))
        ones_2 = np.ones((1, 32, 32, 2))

        val1 = loss_fn(ones_1, 1-ones_1).numpy()
        val2 = loss_fn(ones_2, 1-ones_2).numpy()

        # Values should be scalars
        self.assertEqual(val1.shape, ())
        self.assertEqual(val2.shape, ())

        # Loss values should be equal as they represent the same data, just in different shapes
        self.assertAlmostEqual(val1, val2, 10)

    def test_loss_values(self):
        loss_fn = CategoricalFocalLoss(from_logits=False)

        ones = np.ones((32, 32))
        zeros = np.zeros((32, 32))
        mixed = np.concatenate([ones[:16], zeros[:16]])

        # Predict everything as class 1
        y_pred = np.stack([zeros, ones], axis=-1)

        y_true1 = np.stack([ones, zeros], axis=-1) # All class 0
        y_true2 = np.stack([zeros, ones], axis=-1) # All class 1
        y_true3 = np.stack([mixed, 1-mixed], axis=-1) # Half class 1, half class 0

        # Compute loss values for different labels
        val1 = loss_fn(y_true1, y_pred).numpy() # Should be biggest (all are wrong)
        val2 = loss_fn(y_true2, y_pred).numpy() # Should be 0 (all are correct)
        val3 = loss_fn(y_true3, y_pred).numpy() # Should be in between (half are correct)

        self.assertAlmostEqual(val2, 0.0, 10)

        self.assertGreater(val3, val2)
        self.assertGreater(val1, val3)

if __name__ == '__main__':
    unittest.main()