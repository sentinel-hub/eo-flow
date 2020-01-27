import os
import numpy as np
import tensorflow as tf
from marshmallow import fields, Schema
from marshmallow.validate import OneOf
from eolearn.core import EOPatch, FeatureType

from ..base import BaseInput

_valid_types = [t.value for t in FeatureType]


def eopatch_dataset(root_dir_or_list, features_data, fill_na=None):
    """ Creates a tf dataset with features from saved EOPatches.

    :param root_dir_or_list: Root directory containing eopatches or a list of eopatch directories.
    :type root_dir_or_list: str or list(str)
    :param features_data: List of tuples containing data about features to extract.
        Tuple structure: (feature_type, feature_name, out_feature_name, feature_dtype, feature_shape)
    :type features_data: (str, str, str, np.dtype, tuple)
    :param fill_na: Value with wich to replace nan values. No replacement is done if None.
    :type fill_na: int
    """

    if isinstance(root_dir_or_list, str):
        file_pattern = os.path.join(root_dir_or_list, '*')
        dataset = tf.data.Dataset.list_files(file_pattern)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(root_dir_or_list)

    def _read_patch(path):
        """ TF op for reading an eopatch at a given path. """
        def _func(path):
            path = path.numpy().decode('utf-8')

            # Load only relevant features
            features = [(data[0], data[1]) for data in features_data]
            patch = EOPatch.load(path, features=features)

            data = []
            for feat_type, feat_name, out_name, dtype, shape in features_data:
                arr = patch[feat_type][feat_name].astype(dtype)

                if fill_na is not None:
                    arr[np.isnan(arr)] = fill_na

                data.append(arr)

            return data

        out_types = [tf.as_dtype(data[3]) for data in features_data]
        data = tf.py_function(_func, [path], out_types)

        out_data = {}
        for f_data, feature in zip(features_data, data):
            feat_type, feat_name, out_name, dtype, shape = f_data
            feature.set_shape(shape)
            out_data[out_name] = feature

        return out_data

    dataset = dataset.map(_read_patch)
    return dataset


class EOPatchSegmentationInput(BaseInput):
    """ An input method for basic EOPatch reading. Reads features and segmentation labels. For more complex behaviour
        (subpatch extraction, data augmentation, caching, ...) create your own input method (see examples).
    """

    class _Schema(Schema):
        data_dir = fields.String(description="The directory containing EOPatches.", required=True)

        input_feature_type = fields.String(description="Feature type of the input feature.", required=True, validate=OneOf(_valid_types))
        input_feature_name = fields.String(description="Name of the input feature.", required=True)
        input_feature_axis = fields.List(fields.Int, description="Height and width axis for the input features", required=True, example=[1,2])
        input_feature_shape = fields.List(fields.Int, description="Shape of the input feature. Use -1 for unknown dimesnions.",
                                          required=True, example=[-1, 100, 100, 3])

        labels_feature_type = fields.String(description="Feature type of the labels feature.", required=True, validate=OneOf(_valid_types))
        labels_feature_name = fields.String(description="Name of the labels feature.", required=True)
        labels_feature_axis = fields.List(fields.Int, description="Height and width axis for the labels", required=True, example=[1,2])
        labels_feature_shape = fields.List(fields.Int, description="Shape of the labels feature. Use -1 for unknown dimesnions.",
                                           required=True, example=[-1, 100, 100, 3])

        batch_size = fields.Int(description="Number of examples in a batch.", required=True, example=20)
        num_classes = fields.Int(description="Number of classes. Used for one-hot encoding.", required=True, example=2)

    def _parse_shape(self, shape):
        shape = [None if s < 0 else s for s in shape]
        return shape

    def get_dataset(self):
        cfg = self.config

        # Create a tf.data.Dataset from EOPatches
        features_data = [
            (cfg.input_feature_type, cfg.input_feature_name, 'features', np.float32, self._parse_shape(cfg.input_feature_shape)),
            (cfg.labels_feature_type, cfg.labels_feature_name, 'labels', np.int64, self._parse_shape(cfg.labels_feature_shape))
        ]
        dataset = eopatch_dataset(self.config.data_dir, features_data, fill_na=-2)

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
