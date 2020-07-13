import unittest
import numpy as np

from eoflow.models.losses import CategoricalCrossEntropy, CategoricalFocalLoss
from eoflow.models.losses import JaccardDistanceLoss, TanimotoDistanceLoss


class TestLosses(unittest.TestCase):
    def test_shapes(self):
        for loss_fn in [CategoricalFocalLoss(from_logits=True), CategoricalCrossEntropy(from_logits=True)]:

            ones_1 = np.ones((1, 1024, 2))
            ones_2 = np.ones((1, 32, 32, 2))

            val1 = loss_fn(ones_1, 1-ones_1).numpy()
            val2 = loss_fn(ones_2, 1-ones_2).numpy()

            # Values should be scalars
            self.assertEqual(val1.shape, ())
            self.assertEqual(val2.shape, ())

            # Loss values should be equal as they represent the same data, just in different shapes
            self.assertAlmostEqual(val1, val2, 10)

    def test_focal_loss_values(self):
        ones = np.ones((32, 32))
        zeros = np.zeros((32, 32))
        mixed = np.concatenate([ones[:16], zeros[:16]])

        # Predict everything as class 1
        y_pred = np.stack([zeros, ones], axis=-1)

        y_true1 = np.stack([ones, zeros], axis=-1)  # All class 0
        y_true2 = np.stack([zeros, ones], axis=-1)  # All class 1
        y_true3 = np.stack([mixed, 1-mixed], axis=-1)  # Half class 1, half class 0

        for loss_fn in [CategoricalFocalLoss(from_logits=False),
                        CategoricalFocalLoss(from_logits=False, class_weights=np.array([0, 1]))]:

            # Compute loss values for different labels
            val1 = loss_fn(y_true1, y_pred).numpy()  # Should be biggest (all are wrong)
            val2 = loss_fn(y_true2, y_pred).numpy()  # Should be 0 (all are correct)
            val3 = loss_fn(y_true3, y_pred).numpy()  # Should be in between (half are correct)

            self.assertAlmostEqual(val2, 0.0, 10)

            self.assertGreaterEqual(val3, val2)
            self.assertGreaterEqual(val1, val3)

    def test_jaccard_loss(self):
        loss_fn = JaccardDistanceLoss(from_logits=False, smooth=1)

        y_true = np.zeros([1, 32, 32, 3])
        y_true[:, :16, :16, 0] = np.ones((1, 16, 16))
        y_true[:, 16:, :16, 1] = np.ones((1, 16, 16))
        y_true[:, :, 16:, 2] = np.ones((1, 32, 16))

        y_pred = np.zeros([1, 32, 32, 3])
        y_pred[..., 0] = 1

        val_1 = loss_fn(y_true, y_true).numpy()
        val_2 = loss_fn(y_true, y_pred).numpy()
        y_pred[..., 0] = 0
        y_pred[..., 1] = 1
        val_3 = loss_fn(y_true, y_pred).numpy()
        y_pred[..., 1] = 0
        y_pred[..., 2] = 1
        val_4 = loss_fn(y_true, y_pred).numpy()

        self.assertEqual(val_1, 0.0)
        self.assertAlmostEqual(val_2, 2.743428, 5)
        self.assertAlmostEqual(val_3, 2.743428, 5)
        self.assertAlmostEqual(val_4, 2.491730, 5)

        loss_fn = JaccardDistanceLoss(from_logits=False, smooth=1, class_weights=np.array([0, 1, 1]))

        val_1 = loss_fn(y_true, y_true).numpy()
        val_2 = loss_fn(y_true, y_pred).numpy()
        y_pred[..., 0] = 0
        y_pred[..., 1] = 1
        val_3 = loss_fn(y_true, y_pred).numpy()
        y_pred[..., 1] = 0
        y_pred[..., 2] = 1
        val_4 = loss_fn(y_true, y_pred).numpy()

        self.assertEqual(val_1, 0.0)
        self.assertAlmostEqual(val_2, 1.495621, 5)
        self.assertAlmostEqual(val_3, 1.248781, 5)
        self.assertAlmostEqual(val_4, 1.495621, 5)

    def test_tanimoto_loss(self):

        y_true = np.zeros([1, 32, 32, 2], dtype=np.float32)
        y_true[:, 16:, :16, 1] = np.ones((1, 16, 16))
        y_true[..., 0] = np.ones([1, 32, 32]) - y_true[..., 1]

        y_pred = np.zeros([1, 32, 32, 2], dtype=np.float32)
        y_pred[..., 0] = 1

        self.assertEqual(TanimotoDistanceLoss(from_logits=False)(y_true, y_true).numpy(), 0.0)
        self.assertEqual(TanimotoDistanceLoss(from_logits=False)(y_pred, y_pred).numpy(), 0.0)
        self.assertAlmostEqual(TanimotoDistanceLoss(from_logits=False)(y_true, y_pred).numpy(), 1.25, 5)
        self.assertAlmostEqual(TanimotoDistanceLoss(from_logits=False, normalise=True)(y_true, y_pred).numpy(),
                               1.2460148, 5)
        self.assertAlmostEqual(TanimotoDistanceLoss(from_logits=False, class_weights=np.array([1, 0]))(y_true,
                                                                                                       y_pred).numpy(),
                               0.25, 5)

        y_true = np.zeros([1, 32, 32, 2], dtype=np.float32)
        y_true[..., 0] = np.ones([1, 32, 32]) - y_true[..., 1]
        self.assertEqual(TanimotoDistanceLoss(from_logits=False, normalise=True)(y_true, y_pred).numpy(), 0.)


if __name__ == '__main__':
    unittest.main()