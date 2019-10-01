import os
import numpy as np
import tensorflow as tf
from marshmallow import fields, Schema
from marshmallow.validate import OneOf
from eolearn.core import EOPatch, FeatureType

from ..base import BaseInput

_valid_types = [t.value for t in FeatureType]

def extract_patches(patch_size, spatial_features_and_axis, random_sampling=False, num_random_samples=20, grid_overlap=0.2):
    """ Extract patches from EOPatch dataset. """

    patch_w,patch_h = patch_size
    def _fn(data):
        feat_name_ref, axis_ref = spatial_features_and_axis[0]
        ay_ref, ax_ref = axis_ref
        # Get random coordinates
        def _py_get_random(image):
            x_space = image.shape[ax_ref]-patch_w
            if x_space > 0:
                x_rand = np.random.randint(x_space, size=num_random_samples, dtype=np.int64)
            else:
                x_rand = np.zeros(num_random_samples, np.int64)

            y_space = image.shape[ay_ref]-patch_h 
            if y_space > 0:
                y_rand = np.random.randint(y_space, size=num_random_samples, dtype=np.int64)
            else:
                y_rand = np.zeros(num_random_samples, np.int64)

            return x_rand, y_rand

        # Get coordinates on a grid
        def _py_get_gridded(image):

            # alpha is overlaping ratio (w.r.t. patch size)
            alpha = grid_overlap

            img_height = image.shape[ay_ref]
            img_width = image.shape[ax_ref]

            # number of patches in x and y direction
            nx = int(np.ceil((img_width - alpha * patch_w) / (patch_w * (1 - alpha))))
            ny = int(np.ceil((img_height - alpha * patch_h) / (patch_h * (1 - alpha))))

            # total number of patches
            N = nx * ny
            # allocate output vectors (top-left patch coordinates)
            tl_x = np.zeros(N, dtype=np.int64)
            tl_y = np.zeros(N, dtype=np.int64)

            # calculate actual x and y coordinates
            for yi in range(ny):
                if yi == 0:
                    # the highest patch has x0 = 0
                    y_ = 0
                elif yi == ny - 1:
                    # the lowest patch has y0 = H - patch_h
                    y_ = img_height - patch_h
                else:
                    # calculate top-left y coordinate and take into account overlaping, too
                    y_ = np.round(yi * patch_h - yi * alpha * patch_h)

                for xi in range(nx):
                    if xi == 0:
                        # the left-most patch has x0 = 0
                        x_ = 0
                    elif xi == nx - 1:
                        # the right-most patch has x0 = W - patch_w
                        x_ = img_width - patch_w
                    else:
                        # calculate top-left x coordinate and take into account overlaping, too
                        x_ = np.round(xi * patch_w - xi * alpha * patch_w)

                    id = yi * nx + xi
                    tl_x[id] = np.int64(x_)
                    tl_y[id] = np.int64(y_)

            return tl_x, tl_y

        if random_sampling:
            x_samp, y_samp = tf.py_func(_py_get_random, [data[feat_name_ref]], [tf.int64, tf.int64])
        else:
            x_samp, y_samp = tf.py_func(_py_get_gridded, [data[feat_name_ref]], [tf.int64, tf.int64])

        
        def _py_get_patches(axis):
            ay, ax = axis
            # Extract patches for given coordinates
            def _func(image, x_samp, y_samp):
                patches = []

                # Pad if necessary
                x_pad = max(0, patch_w - image.shape[ax])
                y_pad = max(0, patch_h - image.shape[ay])

                if x_pad > 0 or y_pad > 0:
                    print("Padding...")
                    pad_x1 = x_pad//2
                    pad_x2 = x_pad - pad_x1
                    pad_y1 = y_pad//2
                    pad_y2 = y_pad - pad_y1

                    padding = [(0,0) for _ in range(image.ndim)]
                    padding[ax] = (pad_x1,pad_x2)
                    padding[ay] = (pad_y1,pad_y2)
                    image = np.pad(image, padding, 'constant')
                
                # Extract patches
                print("TEEEEEEEEST", x_samp)
                for x, y in zip(x_samp, y_samp):
                    # Slice on specified axis
                    slicing = [slice(None) for _ in range(image.ndim)]
                    slicing[ax] = slice(x, x+patch_w)
                    slicing[ay] = slice(y, y+patch_h)

                    patch = image[slicing]
                    patches.append(patch)
                return np.stack(patches)

            return _func

        data_out = {}
        # TODO: repeat the rest of the data
        for feat_name, axis in spatial_features_and_axis:
            ay, ax = axis
            shape = data[feat_name].shape.as_list()
            patches = tf.py_func(_py_get_patches(axis), [data[feat_name], x_samp, y_samp], data[feat_name].dtype)
            
            # Update shape information
            shape[ax] = patch_w
            shape[ay] = patch_h
            shape = [None] + shape
            patches.set_shape(shape)

            data_out[feat_name] = patches

        return tf.data.Dataset.from_tensor_slices(data_out)

    return _fn

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

        features.set_shape((None,)*self.config.input_feature_ndims)
        labels.set_shape((None,)*self.config.labels_feature_ndims)

        data = {
            'features': features,
            'labels': labels
        }
        return data

    def get_dataset(self):
        file_pattern = os.path.join(self.config.data_dir, '*')
        dataset = tf.data.Dataset.list_files(file_pattern)

        dataset = dataset.map(self._read_eopatch)

        extract_fn = extract_patches(
            self.config.patch_size,
            [('features', self.config.input_feature_axis),
             ('labels', self.config.labels_feature_axis)],
        )
        dataset = dataset.flat_map(extract_fn)

        return dataset

