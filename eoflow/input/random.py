import numpy as np
import tensorflow as tf
from marshmallow import fields, Schema

from ..base import BaseInput


class RandomClassificationInput(BaseInput):
    """ Class to create random batches for classification tasks. Can be used for prototyping. """

    class _Schema(Schema):
        input_shape = fields.List(fields.Int, description="Shape of a single input example.", required=True, example=[784])
        num_classes = fields.Int(description="Number of classes.", required=True, example=10)

        batch_size = fields.Int(description="Number of examples in a batch.", required=True, example=20)
        batches_per_epoch = fields.Int(required=True, description='Number of batches in epoch', example=20)

    def _generate_batch(self):
        for i in range(self.config.batches_per_epoch):
            input_shape = [self.config.batch_size] + self.config.input_shape
            input_data = np.random.rand(*input_shape)

            onehot = np.eye(self.config.num_classes)
            output_shape = [self.config.batch_size]
            classes = np.random.randint(self.config.num_classes, size=output_shape)
            labels = onehot[classes]

            yield input_data, labels

    def get_dataset(self):
        input_shape = [self.config.batch_size] + self.config.input_shape
        output_shape = [self.config.batch_size, self.config.num_classes]

        dataset = tf.data.Dataset.from_generator(
            self._generate_batch,
            (tf.float32, tf.float32),
            (tf.TensorShape(input_shape), tf.TensorShape(output_shape))
        )

        return dataset


class RandomSegmentationInput(BaseInput):
    """ Class to create random batches for segmentation tasks. Can be used for prototyping. """

    class _Schema(Schema):
        input_shape = fields.List(fields.Int, description="Shape of a single input example.", required=True, example=[512,512,3])
        output_shape = fields.List(fields.Int, description="Shape of a single output mask.", required=True, example=[128,128])
        num_classes = fields.Int(description="Number of segmentation classes.", required=True, example=10)

        batch_size = fields.Int(description="Number of examples in a batch.", required=True, example=20)
        batches_per_epoch = fields.Int(required=True, description='Number of batches in epoch', example=20)

    def _generate_batch(self):
        for i in range(self.config.batches_per_epoch):
            input_shape = [self.config.batch_size] + self.config.input_shape
            input_data = np.random.rand(*input_shape)

            onehot = np.eye(self.config.num_classes)
            output_shape = [self.config.batch_size] + self.config.output_shape
            classes = np.random.randint(self.config.num_classes, size=output_shape)
            labels = onehot[classes]

            yield input_data, labels

    def get_dataset(self):
        input_shape = [self.config.batch_size] + self.config.input_shape
        output_shape = [self.config.batch_size] + self.config.output_shape + [self.config.num_classes]

        dataset = tf.data.Dataset.from_generator(
            self._generate_batch,
            (tf.float32, tf.float32),
            (tf.TensorShape(input_shape), tf.TensorShape(output_shape))
        )

        return dataset
