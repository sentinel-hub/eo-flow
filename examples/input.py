import json

import numpy as np
import tensorflow as tf
from marshmallow import fields, Schema

from eolearn.core import FeatureType
from eoflow.base import BaseInput
from eoflow.input.eopatch import eopatch_dataset, EOPatchSegmentationInput
from eoflow.input.operations import extract_subpatches, augment_data, cache_dataset

_valid_types = [t.value for t in FeatureType]


class ExampleInput(BaseInput):
    """ A simple example of an Input class. Produces random data. """

    class _Schema(Schema):
        input_shape = fields.List(fields.Int, description="Shape of a single input example.", required=True, example=[784])
        num_classes = fields.Int(description="Number of classes.", required=True, example=10)

        batch_size = fields.Int(description="Number of examples in a batch.", required=True, example=20)
        batches_per_epoch = fields.Int(required=True, description='Number of batches in epoch', example=20)

    def _generate_batch(self):
        """ Generator that returns random features and labels. """

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

        # Create a tf dataset from a np.array generator
        dataset = tf.data.Dataset.from_generator(
            self._generate_batch,
            (tf.float32, tf.float32),
            (tf.TensorShape(input_shape), tf.TensorShape(output_shape))
        )

        return dataset


class EOPatchInputExample(BaseInput):
    """ An example input method for EOPatches. Shows feature reading, subpatch extraction, data augmentation,
     caching, batching, etc. """

    # Configuration schema (extended from EOPatchSegmentationInput)
    class _Schema(EOPatchSegmentationInput._Schema):
        # New fields
        patch_size = fields.List(fields.Int, description="Width and height of extracted patches.", required=True, example=[1,2])
        num_subpatches = fields.Int(required=True, description="Number of subpatches extracted by random sampling.", example=5)

        interleave_size = fields.Int(description="Number of eopatches to interleave the subpatches from.", required=True, example=5)
        data_augmentation = fields.Bool(missing=False, description="Use data augmentation on images.")

        cache_file = fields.String(
            missing=None, description="A path to the file where the dataset will be cached. No caching if not provided.", example='/tmp/data')

    @staticmethod
    def _parse_shape(shape):
        shape = [None if s < 0 else s for s in shape]
        return shape

    def get_dataset(self):
        cfg = self.config
        print(json.dumps(cfg, indent=4))

        # Create a tf.data.Dataset from EOPatches
        features_data = [
            (cfg.input_feature_type, cfg.input_feature_name, 'features', np.float32, self._parse_shape(cfg.input_feature_shape)),
            (cfg.labels_feature_type, cfg.labels_feature_name, 'labels', np.int64, self._parse_shape(cfg.labels_feature_shape))
        ]
        dataset = eopatch_dataset(self.config.data_dir, features_data, fill_na=-2)

        # Extract random subpatches
        extract_fn = extract_subpatches(
            self.config.patch_size,
            [('features', self.config.input_feature_axis),
             ('labels', self.config.labels_feature_axis)],
            random_sampling=True,
            num_random_samples=self.config.num_subpatches
        )
        # Interleave patches extracted from multiple EOPatches
        dataset = dataset.interleave(extract_fn, self.config.interleave_size)

        # Cache the dataset so the patch extraction is done only once
        if self.config.cache_file is not None:
            dataset = cache_dataset(dataset, self.config.cache_file)

        # Data augmentation
        if cfg.data_augmentation:
            feature_augmentation = [
                ('features', ['flip_left_right', 'rotate', 'brightness']),
                ('labels', ['flip_left_right', 'rotate'])
            ]
            dataset = dataset.map(augment_data(feature_augmentation))

        # One-hot encode labels and return tuple
        def _prepare_data(data):
            features = data['features']
            labels = data['labels'][..., 0]

            labels_oh = tf.one_hot(labels, depth=self.config.num_classes)

            return features, labels_oh

        dataset = dataset.map(_prepare_data)

        # Create batches
        dataset = dataset.batch(self.config.batch_size)

        return dataset
