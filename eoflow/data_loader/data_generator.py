from random import shuffle, choice
from glob import glob
import logging
import pickle
import os
import concurrent.futures
from tqdm.auto import tqdm
import numpy as np
import tensorflow as tf

from scipy.ndimage.morphology import binary_erosion
from skimage.morphology import disk
from marshmallow import Schema, fields

from .jittering import tasks, jitter_axes_4d

from eoflow.base import BaseInput

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class DataGenerator:
    """ Class to read batches """

    def __init__(self, config):
        self.config = config
        self.patchlets = None
        self.train_set = None
        self.cval_set = None
        self.set_idx = None
        if os.path.exists(os.path.join(self.config.log_dir, self.config.train_cval_log)):
            self.load_train_cval()
        else:
            self.train_cval_split()
        self.workers = config.workers
        print('Loading training data')
        self.train_patchlets = self.load_to_ram_parallel(self.train_set)
        print('Loading cross-validation data')
        self.cval_patchlets = self.load_to_ram_parallel(self.cval_set)

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
        data_dirs = glob(os.path.join(data_dir, data_prefix + '*'))
        if not data_dirs:
            raise ValueError("Error loading data. Either non-existing or empty folder")
        # shuffle in place
        shuffle(data_dirs)
        # split eopatch dir names into train and test
        n_train, n_cval = int(np.round(len(data_dirs) * train_ratio)), int(np.round(len(data_dirs) * cval_ratio))
        self.train_set = data_dirs[:n_train]
        self.cval_set = data_dirs[n_train:n_train + n_cval]
        # pickle lists for reproducibility
        with open(os.path.join(self.config.log_dir, self.config.train_cval_log), 'wb') as f:
            pickle.dump([self.train_set, self.cval_set], f)

    @staticmethod
    def read_one(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)

    def load_to_ram_parallel(self, file_names):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            batches = list(tqdm(executor.map(self.read_one, file_names), total=len(file_names)))
            return batches

    @staticmethod
    def jitter_x_y(data, labels, axes_dict, config):
        # expand first dimension of labels to use the jitter correctly
        labels = labels.reshape(((1,) + tuple(config.lb_size) + (config.n_classes,)))
        # Get sorted identifiers for jittering functions
        jitter_ids = sorted(list(tasks.keys()))
        # Randomly selct one action
        jitter = choice(jitter_ids)
        # jitter data and labels
        data, labels = tasks[jitter](data, axes_dict[jitter]), tasks[jitter](labels, axes_dict[jitter])
        labels = labels.squeeze(axis=0)
        return data, labels

    def get_data(self, patchlets):
        batch_x, batch_y = [], []
        # loop through patchlets
        for patchlet in patchlets:
            tmp_batch_x, tmp_batch_y = patchlet
            batch_x.append(tmp_batch_x)
            batch_y.append(tmp_batch_y)
            del tmp_batch_x, tmp_batch_y
        # return data and labels
        return np.stack(batch_x, axis=0), np.stack(batch_y, axis=0)

    def erode_label(self, label, radius):
        for n_class in range(1, self.config.n_classes):
            label[..., n_class] = binary_erosion(label[..., n_class], structure=disk(radius))
        label[..., 0] = np.where(np.sum(label, axis=-1) == 0, 1, label[..., 0])
        return label

class ExampleDataGenerator(BaseInput):
    """ Class to create random example batches """

    class ClassSchema(Schema):
        input_size = fields.Int(required=True, description='Input size of the model', example=784)
        output_size = fields.Int(required=True, description='Output size of the model', example=10)
        batch_size = fields.Int(required=True, description='Batch size', example=20)
        batches_per_epoch = fields.Int(required=True, description='Number of batches in epoch', example=20)


    def generate(self):
        i_s = [self.config.batch_size, self.config.input_size]
        l_s = [self.config.batch_size, self.config.output_size]

        for i in range(self.config.batches_per_epoch):
            # input data
            input_data = np.random.rand(*i_s)

            # one hot labels
            I = np.eye(self.config.output_size)
            indices = np.random.randint(self.config.output_size, size=self.config.batch_size)
            labels = I[indices]

            yield input_data, labels

    def get_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            self.generate, 
            (tf.float32, tf.int64),
            (tf.TensorShape([None, self.config.input_size]), tf.TensorShape([None, self.config.output_size]))
        )

        return dataset

class MultiTempBatchGenerator(DataGenerator):
    def __init__(self, config):
        super(MultiTempBatchGenerator, self).__init__(config)

    def next_batch(self, batch_size, is_training=True):
        # training or testing
        self.patchlets = self.train_patchlets if is_training else self.cval_patchlets
        if self.patchlets:
            # select batches randomly or loop through datasets
            self.set_idx = np.random.choice(len(self.patchlets), batch_size)
            batch_files = [self.patchlets[ii] for ii in self.set_idx]
            if self.config.jitter:
                batch_files = [self.jitter_x_y(*patchlet, jitter_axes_4d, self.config) for patchlet in batch_files]
            batch_x, batch_y = self.get_data(batch_files)
            if self.config.erode_labels is not None:
                eroded = [self.erode_label(label, self.config.erode_labels) for label in batch_y]
                batch_y = np.stack(eroded, axis=0)
            yield batch_x, batch_y
        else:
            yield None, None
