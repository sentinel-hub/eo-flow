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

def _read_numpy_file(file_path, fields):
    """ Reads a single npz file. """

    data = np.load(file_path)
    np_arrays = [data[f] for f in fields]

    # Check that arrays match in the first dimension
    n_samples = np_arrays[0].shape[0]
    assert all(n_samples == arr.shape[0] for arr in np_arrays)

    return tuple(np_arrays)

def npz_dir_dataset(file_dir_or_list, features, num_parallel=5):
    """ Creates a tf.data.Dataset from a directory containing numpy .npz files. Files are loaded
    lazily when needed. `num_parallel` files are read in parallel and interleaved together.

    :param file_dir_or_list: directory containing .npz files or a list of paths to .npz files
    :type file_dir_or_list: str | list(str)
    :param features: dict of (`field` -> `feature_name`) mappings, where `field` is the field in the .npz array
                   and `feature_name` is the name of the feature it is saved to.
    :type features: dict
    :param num_parallel: number of files to read in parallel and intereleave, defaults to 5
    :type num_parallel: int, optional

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
    shapes = tuple((None,) + arr.shape[1:] for arr in np_arrays)

    def _data_generator(files, fields):
        """ Returns samples from one file at a time. """
        for f in files:
            yield _read_numpy_file(f, fields)

    # Converts a database of tuples to database of dicts
    def _to_dict(*features):
        return {name: feat for name, feat in zip(feature_names, features)}

    # Create dataset
    ds = tf.data.Dataset.from_generator(lambda:_data_generator(files, fields), types, shapes)

    # Prefetch needed amount of files for interleaving
    ds = ds.prefetch(num_parallel)

    # Unbatch and interleave
    ds = ds.interleave(lambda *x: tf.data.Dataset.from_tensor_slices(x), cycle_length=num_parallel)
    ds = ds.map(_to_dict)

    return ds
