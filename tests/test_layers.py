import unittest

import numpy as np
import tensorflow as tf
from eoflow.models.layers import ResConv2D, PyramidPoolingModule


class TestLayers(unittest.TestCase):
    def test_res_conv_layer(self):
        input_shape = (4, 28, 28, 3)

        x = tf.ones(input_shape)

        for npar in range(1, 4):
            y = ResConv2D(3, kernel_size=[npar]*npar, num_parallel=npar, padding='SAME')(x)
            self.assertEqual(y.shape, input_shape)

            y = ResConv2D(3, kernel_size=[npar]*npar, dilation=[npar]*npar, num_parallel=npar, padding='SAME')(x)
            self.assertEqual(y.shape, input_shape)

            y = ResConv2D(3, dilation=[npar]*npar, num_parallel=npar, padding='SAME')(x)
            self.assertEqual(y.shape, input_shape)

        with self.assertRaises(ValueError):
            ResConv2D(filters=3, kernel_size=[3, 3], padding='SAME', num_parallel=3)
            ResConv2D(filters=3, dilation=[3, 3], padding='SAME', num_parallel=3)

    def test_ppm_layer(self):
        batch_size, height, width, nchannels = 1, 64, 64, 1
        input_shape = (batch_size, height, width, nchannels)
        filters = 4
        bins = (1, 2, 4, 8)

        x = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)

        ppm = PyramidPoolingModule(filters=filters, bins=bins, interpolation='nearest')

        y = ppm(x)

        self.assertEqual(y.shape, (batch_size, height, width, filters+nchannels))
        np.testing.assert_array_equal(y[..., 0], x[..., 0])
        for nbin, bin_size in enumerate(bins):
            self.assertLessEqual(np.unique(y[..., nbin+1]).size, bin_size**2)


if __name__ == '__main__':
    unittest.main()
