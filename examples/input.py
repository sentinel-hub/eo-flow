import numpy as np
import tensorflow as tf
from marshmallow import fields, Schema
from marshmallow.validate import OneOf

from eolearn.core import FeatureType
from eoflow.base import BaseInput
from eoflow.input.eopatch import eopatch_dataset, EOPatchSegmentationInput
from eoflow.input.operations import extract_subpatches, augment_data, cache_dataset

_valid_types = [t.value for t in FeatureType]

class EOPatchInputExample(BaseInput):
    """ An example input method. Shows reading EOPatches, subpatch extraction, data augmentation, caching, batching, etc. """

    # Configuration schema (extended from EOPatchSegmentationInput)
    class _Schema(EOPatchSegmentationInput._Schema):
        # New fields
        patch_size = fields.List(fields.Int, description="Width and height of extracted patches.", required=True, example=[1,2])
        num_subpatches = fields.Int(required=True, description="Number of subpatches extracted by random sampling.", example=5)

        interleave_size = fields.Int(description="Number of eopatches to interleave the subpatches from.", required=True, example=5)

        cache_file = fields.String(
            missing=None, description="A path to the file where the dataset will be cached. No caching if not provided.", example='/tmp/data')

    def _parse_shape(self, shape):
        shape = [None if s<0 else s for s in shape]
        return shape

    def get_dataset(self):
        cfg = self.config
        print(cfg)

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
        feature_augmentation = [
            ('features', ['flip_left_right', 'rotate', 'brightness']),
            ('labels', ['flip_left_right', 'rotate'])
        ]
        dataset = dataset.map(augment_data(feature_augmentation))

        # One-hot encode labels and return tuple
        def _prepare_data(data):
            features = data['features']
            labels = data['labels'][...,0]

            labels_oh = tf.one_hot(labels, depth=self.config.num_classes)

            return features, labels_oh

        dataset = dataset.map(_prepare_data)

        # Create batches
        dataset = dataset.batch(self.config.batch_size)

        return dataset
