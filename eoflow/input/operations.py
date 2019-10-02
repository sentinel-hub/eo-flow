import os
import numpy as np
import tensorflow as tf


def extract_subpatches(patch_size, spatial_features_and_axis, random_sampling=False, num_random_samples=20, grid_overlap=0.2):
    """ Builds a TF op for building a dataset of subpatches from tensors. Subpatches sampling can be random or grid based. """

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
                    pad_x1 = x_pad//2
                    pad_x2 = x_pad - pad_x1
                    pad_y1 = y_pad//2
                    pad_y2 = y_pad - pad_y1

                    padding = [(0,0) for _ in range(image.ndim)]
                    padding[ax] = (pad_x1,pad_x2)
                    padding[ay] = (pad_y1,pad_y2)
                    image = np.pad(image, padding, 'constant')

                # Extract patches
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

        # TODO: shuffle subpatches
        return tf.data.Dataset.from_tensor_slices(data_out)

    return _fn

def augment_data(features_to_augment, brightness_delta=0.1, contrast_bounds=(0.9,1.1)):
    """ Builds a function that randomly augments features in specified ways. """
    
    def _augment(data):
        contrast_lower, contrast_upper = contrast_bounds

        flip_lr_cond = tf.random.uniform(shape=[]) > 0.5
        flip_ud_cond = tf.random.uniform(shape=[]) > 0.5
        rot90_amount = tf.random.uniform(shape=[], maxval=4, dtype=tf.int32)

        # Available operations
        operations = {
            'flip_left_right': lambda x: tf.cond(flip_lr_cond, lambda: tf.image.flip_left_right(x), lambda: x),
            'flip_up_down': lambda x: tf.cond(flip_ud_cond, lambda: tf.image.flip_up_down(x), lambda: x),
            'rotate': lambda x: tf.image.rot90(x, rot90_amount),
            'brightness': lambda x: tf.image.random_brightness(x, brightness_delta),
            'contrast': lambda x: tf.image.random_contrast(x, contrast_lower, contrast_upper)
        }

        for feature, ops in features_to_augment:
            # Apply specified ops to feature
            for op in ops:
                operation_fn = operations[op]
                data[feature] = operation_fn(data[feature])
        
        return data

    return _augment