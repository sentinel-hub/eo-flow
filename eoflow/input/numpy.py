import os

import numpy as np
import tensorflow as tf

def numpy_dataset(np_array_dict):
    """ Creates a tf.data Dataset from a dict of numpy arrays. """

    # Unpack
    feature_names = list(np_array_dict.keys())
    np_arrays = [np_array_dict[name] for name in feature_names]

    # Check that arrays match in the first dimension
    n_samples = np_arrays[0].shape[0]
    assert all(n_samples == arr.shape[0] for arr in np_arrays)

    # Extract types and shapes form np arrays
    types = tuple(arr.dtype for arr in np_arrays)
    shapes = tuple(arr.shape[1:] for arr in np_arrays)

    def _generator():
        # Iterate through the first dimension of arrays
        for slices in zip(*np_arrays):
            yield slices

    ds = tf.data.Dataset.from_generator(_generator, types, shapes)
    ds = ds.take(n_samples)

    # Converts a database of tuples to database of dicts
    def _to_dict(*features):
        return {name: feat for name, feat in zip(feature_names, features)}

    ds = ds.map(_to_dict)

    return ds

def npz_dir_dataset(file_dir_or_list, features, num_parallel=5, shuffle_size=100):
    """ Creates a tf.data.Dataset from a directory containing numpy .npz files.

    :param file_dir_or_list: directory containing .npz files or a list of paths to .npz files
    :type file_dir_or_list: str | list(str)
    :param features: dict of (`field` -> `feature_name`) mappings, where `field` is the field in the .npz array
                   and `feature_name` is the name of the feature it is saved to.
    :type features: dict
    :param num_parallel: number of files to read in parallel and intereleave, defaults to 5
    :type num_parallel: int, optional
    :param shuffle_size: buffer size for shuffling file order, defaults to 100
    :type shuffle_size: int, optional

    :return: dataset containing examples merged from files
    :rtype: tf.data.Dataset
    """

    files = file_dir_or_list

    # If dir, then list files
    if isinstance(file_dir_or_list, str):
        files = [os.path.join(file_dir_or_list, f) for f in os.listdir(file_dir_or_list)]

    fields = list(features.keys())
    feature_names = [features[f] for f in features]

    # Read one file for shape info
    file = next(iter(files))
    data = np.load(file)
    np_arrays = [data[f] for f in fields]

    # Read shape and type info
    types = tuple(arr.dtype for arr in np_arrays)
    shapes = tuple(arr.shape[1:] for arr in np_arrays)

    # Create datasets
    datasets = [_npz_file_lazy_dataset(file, fields, feature_names, types, shapes) for file in files]
    ds = tf.data.Dataset.from_tensor_slices(datasets)

    # Shuffle files and interleave multiple files in parallel
    ds = ds.shuffle(shuffle_size)
    ds = ds.interleave(lambda x:x, cycle_length=num_parallel)

    return ds


def _npz_file_lazy_dataset(file_path, fields, feature_names, types, shapes):
    """ Creates a lazy tf.data Dataset from a numpy file.
    Reads the file when first consumed.

    :param file_path: path to the numpy file
    :type file_path: str
    :param fields: fields to read from the numpy file
    :type fields: list(str)
    :param feature_names: feature names assigned to the fields
    :type feature_names: list(str)
    :param types: types of the numpy fields
    :type types: list(np.dtype)
    :param shapes: shapes of the numpy fields
    :type shapes: list(tuple)

    :return: dataset containing examples from the file
    :rtype: tf.data.Dataset
    """


    def _generator():
        data = np.load(file_path)
        np_arrays = [data[f] for f in fields]

        # Check that arrays match in the first dimension
        n_samples = np_arrays[0].shape[0]
        assert all(n_samples == arr.shape[0] for arr in np_arrays)

        # Iterate through the first dimension of arrays
        for slices in zip(*np_arrays):
            yield slices

    ds = tf.data.Dataset.from_generator(_generator, types, shapes)

    # Converts a database of tuples to database of dicts
    def _to_dict(*features):
        return {name: feat for name, feat in zip(feature_names, features)}

    ds = ds.map(_to_dict)

    return ds
