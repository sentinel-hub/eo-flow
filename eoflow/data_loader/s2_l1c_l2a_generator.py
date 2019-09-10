from random import choice
import numpy as np
import logging

from .jittering import tasks, jitter_axes_3d

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class OnlineEODataGenerator:
    """ Class to read EOPatches """
    def __init__(self, config, workflow):
        self.config = config
        self.workflow = workflow

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

    def _check_dimensionality(self, eopatch):
        data = eopatch.data[self.config.l1c_field]
        labels = eopatch.data[self.config.l2a_field]
        # size of image data and labels
        _, data_h, data_w, data_d = data.shape
        _, labels_h, labels_w, labels_d = labels.shape
        image_h, image_w, image_d = self.config.im_size
        lb_h, lb_w, lb_d = self.config.lb_size
        # check if dimensions agree (i.e. same number of channels, and input heigh and width are large enough)
        if not (data_d == image_d and data_h >= image_h and data_w >= image_w):
            raise ValueError("Specified image size and actual image size do not agree")
        # size of cropping applied by network, determined by number of layers and convolutional kernel size
        crop = self.config.spatial_cropping  # self.compute_cropping(self.config.n_layers, self.config.conv_size)
        # check dimensions of images and labels agree
        if not (data_h == labels_h and data_w == labels_w):
            raise ValueError("Input image and label image sizes must agree")
        # check required dimensions agree with cropping size
        if not (lb_h + crop == image_h) or not (lb_w + crop == image_w):
            raise ValueError("Wrong image/label/CROP variables initialisation")

    def _crop_data_and_labels(self, eopatch):
        data = eopatch.data[self.config.l1c_field].squeeze()
        labels = eopatch.data[self.config.l2a_field].squeeze()
        # size of cropping applied by network, determined by number of layers and convolutional kernel size
        crop = self.config.spatial_cropping  # self.compute_cropping(self.config.n_layers, self.config.conv_size)
        # size of image data and labels
        data_h, data_w, data_d = data.shape
        image_h, image_w, image_d = self.config.im_size
        lb_h, lb_w, lb_d = self.config.lb_size
        # crop images and labels starting from random point
        h_pad, w_pad = data_h - image_h, data_w - image_w
        rand_h_pad = np.random.choice(h_pad, 1)[0]
        rand_w_pad = np.random.choice(w_pad, 1)[0]
        data = data[rand_h_pad:image_h + rand_h_pad, rand_w_pad:rand_w_pad + image_w, :]
        labels = labels[rand_h_pad + crop // 2:rand_h_pad + crop // 2 + lb_h,
                        rand_w_pad + crop // 2:rand_w_pad + crop // 2 + lb_w, :]
        return data, labels

    def get_data(self, batch_size, is_training=True):
        batch_x, batch_y = [], []
        valid_executions = 0
        while valid_executions < batch_size:
            eop, executed = self.workflow.execute()
            if executed:
                self._check_dimensionality(eop)
                data, labels = self._crop_data_and_labels(eop)
                # jitter
                if self.config.jitter and is_training:
                    # compute jitter on the 3d arrays
                    data, labels = self.jitter_x_y(data, labels, jitter_axes_3d)
                data = self.clip_intensities(data)
                labels = self.clip_intensities(labels)
                batch_x.append(data)
                batch_y.append(labels)
                valid_executions += 1
        return np.array(batch_x), np.array(batch_y)

    def next_batch(self, batch_size, is_training=True):
        batch_x, batch_y = self.get_data(batch_size, is_training=is_training)
        yield batch_x.astype(np.float32), batch_y.astype(np.float32)
