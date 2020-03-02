import os

import h5py
import tensorflow as tf

def hdf5_dataset(path, features):
    """ Creates a tf.data.Dataset from a hdf5 file

    :param path: path to the hdf5 file,
    :type path: str
    :param features: dict of (`dataset` -> `feature_name`) mappings, where `dataset` is the dataset name in the hdf5 file
                   and `feature_name` is the name of the feature it is saved to.
    :type features: dict

    :return: dataset containing examples merged from files
    :rtype: tf.data.Dataset
    """

    fields = list(features.keys())
    feature_names = [features[f] for f in features]

    # Reads dataset row by row
    def _generator():
        with h5py.File(path, 'r') as file:
            datasets = [file[field] for field in fields]
            for row in zip(*datasets):
                yield row

    # Converts a database of tuples to database of dicts
    def _to_dict(*features):
        return {name: feat for name, feat in zip(feature_names, features)}

    # Reads hdf5 metadata (types and shapes)
    with h5py.File(path, 'r') as file:
        datasets = [file[field] for field in fields]

        types = tuple(ds.dtype for ds in datasets)
        shapes = tuple(ds.shape[1:] for ds in datasets)

    # Create dataset
    ds = tf.data.Dataset.from_generator(_generator, types, shapes)
    ds = ds.map(_to_dict)

    return ds
