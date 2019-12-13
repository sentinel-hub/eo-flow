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
