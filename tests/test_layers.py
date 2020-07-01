import unittest

import tensorflow as tf
from eoflow.models.layers import ResConv2D


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


if __name__ == '__main__':
    unittest.main()
