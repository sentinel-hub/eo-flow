import os
import numpy as np
import tensorflow as tf
from marshmallow import fields, Schema
from marshmallow.validate import OneOf
from eolearn.core import EOPatch, FeatureType

from ..base import BaseInput

_valid_types = [t.value for t in FeatureType]

class EOPatchInput(BaseInput):
    """ Class to create random batches for classification tasks. """

    class _Schema(Schema):
        data_dir = fields.String(description="The directory containing EOPatches.", required=True)

        input_feature_type = fields.String(description="Feature type of the input feature.", required=True, validate=OneOf(_valid_types))
        input_feature_name = fields.String(description="Name of the input feature.", required=True)

        labels_feature_type = fields.String(description="Feature type of the labels feature.", required=True, validate=OneOf(_valid_types))
        labels_feature_name = fields.String(description="Name of the labels feature.", required=True)

        batch_size = fields.Int(description="Number of examples in a batch.", required=True, example=20)

    def _read_eopatch(self, path):
        """ Reads a features and labels from a single EOPatch. """

        def _func(path):
            path = path.decode('utf-8')
            features = [(self.config.input_feature_type, self.config.input_feature_name),
                        (self.config.labels_feature_type, self.config.labels_feature_name)]
            patch = EOPatch.load(path, features=features)

            input_data = patch[self.config.input_feature_type][self.config.input_feature_name].astype(np.float32)
            labels_data = patch[self.config.labels_feature_type][self.config.labels_feature_name].astype(np.int64)

            return input_data, labels_data

        features, labels = tf.py_func(_func, [path], (tf.float32, tf.int64))

        return features, labels

    def get_dataset(self):
        file_pattern = os.path.join(self.config.data_dir, '*')
        dataset = tf.data.Dataset.list_files(file_pattern)

        dataset = dataset.map(self._read_eopatch)

        return dataset

