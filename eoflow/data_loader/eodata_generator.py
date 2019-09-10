from random import shuffle, choice
from glob import glob
import numpy as np
import logging
import pickle
import os
import re
from datetime import datetime

from eolearn.core import EOPatch

from .jittering import tasks, jitter_axes_3d, jitter_axes_4d

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class BaseEODataGenerator:
    """ Class to read EOPatches """
    def __init__(self, config):
        self.config = config
        self.data_set = None
        self.train_set = None
        self.cval_set = None
        self.set_idx = None
        if os.path.exists(os.path.join(self.config.log_dir, self.config.train_cval_log)):
            self.load_train_cval()
        else:
            self.train_cval_split()
        self.counter = 0

    def load_train_cval(self):
        logging.debug("Loading existing train/test split")
        with open(os.path.join(self.config.log_dir, self.config.train_cval_log), 'rb') as f:
            self.train_set, self.cval_set = pickle.load(f)

    def train_cval_split(self):
        logging.debug("Creating train/test sets split")
        # data with EOPatch folders
        data_dir = self.config.data_dir
        # prefix name of eopatch folders
        data_prefix = self.config.data_prefix
        # train/test split ratio. Check sum is lower than 1
        train_ratio = self.config.train_ratio
        cval_ratio = self.config.cval_ratio
        if train_ratio + cval_ratio > 1:
            raise ValueError("Wrong train/test split ratios")
        # read data folders
        data_dirs = glob(os.path.join(data_dir, data_prefix + '*')) \
            if self.config.eopatches_list is None \
            else [data_dir+os.sep+eop_name for eop_name in pickle.load(open(self.config.eopatches_list, 'rb'))]
        if not data_dirs:
            raise ValueError("Error loading data. Either non-existing or empty folder")
        # shuffle in place
        shuffle(data_dirs)
        # split eopatch dir names into train and test
        n_train, n_cval = int(np.round(len(data_dirs) * train_ratio)), int(np.round(len(data_dirs) * cval_ratio))
        self.train_set = data_dirs[:n_train]
        self.cval_set = data_dirs[n_train:n_train+n_cval]
        # pickle lists for reproducibility
        with open(os.path.join(self.config.log_dir, self.config.train_cval_log), 'wb') as f:
            pickle.dump([self.train_set, self.cval_set], f)

    def load_data(self, eopatch):
        """ load data from eopatch if field exists """
        if self.config.data_field not in eopatch[self.config.data_type]:
            raise ValueError("Data field {:s} not in eopatch".format(self.config.data_field))
        # return data in correct format
        if not self.config.temporal_index:
            if not self.config.band_indices:
                data = self.clip_intensities(eopatch[self.config.data_type][self.config.data_field])
            else:
                data = self.clip_intensities(eopatch[self.config.data_type][self.config.data_field]
                                             [..., self.config.band_indices])
        else:
            if not self.config.band_indices:
                data = self.clip_intensities(eopatch[self.config.data_type][self.config.data_field]
                                             [self.config.temporal_index])
            else:
                data = self.clip_intensities(eopatch[self.config.data_type][self.config.data_field]
                                             [self.config.temporal_index][..., self.config.band_indices])
        if isinstance(self.config.replace_nans, float):
            data[np.isnan(data)] = self.config.replace_nans
        return data

    def load_labels(self, eopatch):
        return eopatch[self.config.mask_type][self.config.mask_field] \
            if self.config.mask_field in eopatch[self.config.mask_type] else None

    def next_batch(self, batch_size, is_training=True, random=True):
        # training or testing
        self.data_set = self.train_set if is_training else self.cval_set
        if self.data_set:
            # select batches randomly or loop through datasets
            if random:
                self.set_idx = np.random.choice(len(self.data_set), batch_size)
            else:
                self.set_idx = np.ogrid[self.counter:self.counter + batch_size]
                self.counter += batch_size
            batch_files = [self.data_set[ii] for ii in self.set_idx]
            batch_x, batch_y = self.get_data(batch_files)
            yield batch_x, batch_y
        else:
            yield None, None

    def get_batch_filenames(self):
        return [self.data_set[ii] for ii in self.set_idx]

    def get_data(self, data_set):
        batch_x, batch_y = [], []
        # loop through filenames
        for filename in data_set:
            tmp_batch_x, tmp_batch_y = self.read_eopatch(filename)
            batch_x.append(tmp_batch_x)
            batch_y.append(tmp_batch_y)
            del tmp_batch_x, tmp_batch_y
        # return data and labels
        return np.stack(batch_x, axis=0), np.stack(batch_y, axis=0)

    def read_eopatch(self, filename):
        raise NotImplementedError

    def one_hot_encoding(self, labels):
        labels = np.squeeze(labels)
        # should be 2d array at this point
        height, width = labels.shape
        new_labels = np.empty((height, width, self.config.n_classes), dtype=np.uint8)
        for cc in np.arange(self.config.n_classes):
            new_labels[:, :, cc] = np.uint8(labels == cc)
        return new_labels

    def clip_intensities(self, x):
        return np.clip(x, self.config.clip_low, self.config.clip_high)

    @staticmethod
    def jitter_x_y(data, labels, axes_dict):
        # Get sorted identifiers for jittering functions
        jitter_ids = sorted(list(tasks.keys()))
        # Randomly selct one action
        jitter = choice(jitter_ids)
        # jitter data and labels
        return tasks[jitter](data, axes_dict[jitter]), tasks[jitter](labels, axes_dict[jitter])


class EOMonoTempDataGenerator(BaseEODataGenerator):
    def __init__(self, config):
        super(EOMonoTempDataGenerator, self).__init__(config)

    def read_eopatch(self, filename):
        # load eopatch file
        eopatch = EOPatch.load(filename)
        # extract data from eopatch
        data = self.load_data(eopatch)
        labels = self.load_labels(eopatch)
        if labels is None:
            raise ValueError("Labels {:s} in {:s} not found".format(self.config.mask_type, self.config.mask_field))
        # one-hot encoding if not already
        if labels.ndim == 2 or (labels.ndim == 3 and labels.shape[-1] == 1):
            labels = self.one_hot_encoding(labels)
        # size of image data and labels
        data_h, data_w, data_d = data.shape
        labels_h, labels_w, labels_d = labels.shape
        image_h, image_w, image_d = self.config.im_size
        lb_h, lb_w = self.config.lb_size
        # check if dimensions agree (i.e. same number of channels, and input heigh and width are large enough)
        if not (data_d == image_d and data_h >= image_h and data_w >= image_w):
            raise ValueError("Specified image size and actual image size do not agree")
        # size of cropping applied by network, determined by number of layers and convolutional kernel size
        crop = self.config.spatial_cropping     # self.compute_cropping(self.config.n_layers, self.config.conv_size)
        # check dimensions of images and labels agree
        if not (data_h == labels_h and data_w == labels_w):
            raise ValueError("Input image and label image sizes must agree")
        # check required dimensions agree with cropping size
        if not (lb_h + crop == image_h) or not (lb_w + crop == image_w):
            raise ValueError("Wrong image/label/CROP variables initialisation")
        # crop images and labels starting from random point
        h_pad, w_pad = data_h - image_h, data_w - image_w
        rand_h_pad = np.random.choice(h_pad, 1)[0]
        rand_w_pad = np.random.choice(w_pad, 1)[0]
        data = data[rand_h_pad:image_h + rand_h_pad, rand_w_pad:rand_w_pad + image_w, :]
        labels = labels[rand_h_pad + crop // 2:rand_h_pad + crop // 2 + lb_h,
                        rand_w_pad + crop // 2:rand_w_pad + crop // 2 + lb_w, :]
        # jitter
        if self.config.jitter:
            # compute jitter on the 3d arrays
            data, labels = self.jitter_x_y(data, labels, jitter_axes_3d)
        return data.astype(np.float32), labels.astype(np.uint8)


class EOMultiTempDataGenerator(BaseEODataGenerator):
    def __init__(self, config):
        super(EOMultiTempDataGenerator, self).__init__(config)

    @staticmethod
    def get_seed_for_patch(filename, shift):
        patch_name = filename.split('/')[-1]
        numbers = [int(s) for s in re.findall(r'\d+', patch_name)]
        if len(numbers) > 2:
            idx, col, row = numbers
        else:
            col, row = numbers
        seed = int((col + row) * (col + row + 1) + 2 * row + shift)
        return seed

    def read_eopatch(self, filename, seed=None, eopatch=None):
        # load eopatch file
        np.random.seed(seed)
        eopatch = EOPatch.load(filename) if eopatch is None else eopatch
        # extract data from eopatch
        data = self.load_data(eopatch)
        labels = self.load_labels(eopatch)
        if labels is None:
            raise ValueError("Labels {:s} in {:s} not found".format(self.config.mask_type, self.config.mask_field))
        # one-hot encoding if not already
        if labels.ndim == 2 or (labels.ndim == 3 and labels.shape[-1] == 1):
            labels = self.one_hot_encoding(labels)
        # size of image data and labels
        data_t, data_h, data_w, data_d = data.shape
        labels_h, labels_w, labels_d = labels.shape
        image_t, image_h, image_w, image_d = self.config.im_size
        lb_h, lb_w = self.config.lb_size
        # check if dimensions agree (i.e. same number of channels, and input height and width are large enough)
        if not (data_d == image_d and data_t == image_t and data_h >= image_h and data_w >= image_w):
            raise ValueError("Specified image size and actual image size do not agree")
        # size of cropping applied by network, determined by number of layers and convolutional kernel size
        crop = self.config.spatial_cropping     # self.compute_cropping(self.config.n_layers, self.config.conv_size)
        # check dimensions of images and labels agree
        if not (data_h == labels_h and data_w == labels_w):
            raise ValueError("Input image and label image sizes must agree")
        # check required dimensions agree with cropping size
        if not (lb_h + crop == image_h) or not (lb_w + crop == image_w):
            raise ValueError("Wrong image/label/CROP variables initialisation")
        # crop images and labels starting from random point
        h_pad, w_pad = data_h - image_h, data_w - image_w
        rand_h_pad = np.random.choice(h_pad, 1)[0]
        rand_w_pad = np.random.choice(w_pad, 1)[0]
        data = data[:, rand_h_pad:image_h + rand_h_pad,
                    rand_w_pad:rand_w_pad + image_w, :]
        labels = labels[rand_h_pad + crop // 2:rand_h_pad + crop // 2 + lb_h,
                        rand_w_pad + crop // 2:rand_w_pad + crop // 2 + lb_w, :]
        # jitter
        if self.config.jitter:
            # expand first dimension to use the jitter correctly
            labels = labels.reshape(((1,) + tuple(self.config.lb_size) + (self.config.n_classes,)))
            # compute jitter on 4d arrays
            data, labels = self.jitter_x_y(data, labels, jitter_axes_4d)
            # squeeze back labels
            labels = labels.squeeze(axis=0)
        if np.sum(labels[..., 0])/(lb_h*lb_w) >= self.config.max_no_data_ratio:
            return None
        return data.astype(np.float32), labels.astype(np.uint8)


class PredictDataGenerator(BaseEODataGenerator):
    def __init__(self, config):
        super(PredictDataGenerator, self).__init__(config)
        self.load_test_set()

    def load_test_set(self):
        test_dir = self.config.test_dir
        # prefix name of eopatch folders
        data_prefix = self.config.data_prefix
        # read test data folders
        data_dirs = glob(os.path.join(test_dir, data_prefix + '*'))
        if not data_dirs:
            raise ValueError("Error loading data. Either non-existing or empty folder")
        self.cval_set = data_dirs

    def read_eopatch(self, filename):
        # load eopatch file
        eopatch = EOPatch.load(filename, lazy_loading=True)
        # extract data from eopatch
        data = self.load_data(eopatch)
        labels = self.load_labels(eopatch)
        # one-hot encoding if not already
        if labels is not None and (labels.ndim == 2 or (labels.ndim == 3 and labels.shape[-1] == 1)):
            labels = self.one_hot_encoding(labels)
        return data.astype(np.float32), labels.astype(np.uint8)
