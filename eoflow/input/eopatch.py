import os
import numpy as np
import tensorflow as tf
from marshmallow import fields, Schema
from marshmallow.validate import OneOf
from eolearn.core import EOPatch, FeatureType

from ..base import BaseInput
from .operations import extract_subpatches

_valid_types = [t.value for t in FeatureType]

def eopatch_dataset(data_dir, features_data):
        """ Reads a features and labels from a single EOPatch.

        :param data_dir: Root directory containing eopatches in the dataset
        :type data_dir: str
        :param features_data: List of tuples containing data about features to extract.
            Tuple structure: (feature_type, feature_name, out_feature_name, feature_dtype, feature_ndims)
        :type features_data: (str, str, str, np.dtype, int)
        """

        file_pattern = os.path.join(data_dir, '*')
        dataset = tf.data.Dataset.list_files(file_pattern)

        def _read_patch(path):
            """ TF op for reading an eopatch at a given path. """
            def _func(path):
                path = path.decode('utf-8')

                # Load only relevant features
                features = [(data[0], data[1]) for data in features_data]
                patch = EOPatch.load(path, features=features)

                data = []
                for feat_type, feat_name, out_name, dtype, ndims in features_data:
                    arr = patch[feat_type][feat_name].astype(dtype)
                    data.append(arr)

                return data

            out_types = [tf.as_dtype(data[3]) for data in features_data]
            data = tf.py_func(_func, [path], out_types)

            out_data = {}
            for f_data, feature in zip(features_data, data):
                feat_type, feat_name, out_name, dtype, ndims = f_data
                feature.set_shape((None,) * ndims)
                out_data[out_name] = feature

            return out_data

        dataset = dataset.map(_read_patch)
        return dataset


class EOPatchInput(BaseInput):
    """ Class to create random batches for classification tasks. """

    class _Schema(Schema):
        data_dir = fields.String(description="The directory containing EOPatches.", required=True)

        input_feature_type = fields.String(description="Feature type of the input feature.", required=True, validate=OneOf(_valid_types))
        input_feature_name = fields.String(description="Name of the input feature.", required=True)
        input_feature_axis = fields.List(fields.Int, description="Height and width axis for the input features", required=True, example=[1,2])
        input_feature_ndims = fields.Int(description="Number of dimensions for the input features", required=True, example=4)

        labels_feature_type = fields.String(description="Feature type of the labels feature.", required=True, validate=OneOf(_valid_types))
        labels_feature_name = fields.String(description="Name of the labels feature.", required=True)
        labels_feature_axis = fields.List(fields.Int, description="Height and width axis for the labels", required=True, example=[1,2])
        labels_feature_ndims = fields.Int(description="Number of dimensions for the labels", required=True, example=3)

        patch_size = fields.List(fields.Int, description="Width and height of extracted patches.", required=True, example=[1,2])

        interleave_size = fields.Int(description="Number of eopatches to interleave the subpatches from.", required=True, example=5)
        batch_size = fields.Int(description="Number of examples in a batch.", required=True, example=20)

    def get_dataset(self):
        cfg = self.config
        features_data = [
            (cfg.input_feature_type, cfg.input_feature_name, 'features', np.float32, cfg.input_feature_ndims),
            (cfg.labels_feature_type, cfg.labels_feature_name, 'labels', np.int64, cfg.labels_feature_ndims)
        ]

        dataset = eopatch_dataset(self.config.data_dir, features_data)

        extract_fn = extract_subpatches(
            self.config.patch_size,
            [('features', self.config.input_feature_axis),
             ('labels', self.config.labels_feature_axis)],
        )
        dataset = dataset.interleave(extract_fn, self.config.interleave_size)

        dataset = dataset.batch(self.config.batch_size)

        return dataset

