import os
import logging
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from marshmallow import Schema, fields
from marshmallow.validate import OneOf, ContainsOnly

from ..base import BaseModel
from .layers import Conv2D, Deconv2D, CropAndConcat
from ..utils.tf_utils import plot_to_image

import types

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# Available losses. Add keys with new losses here.
segmentation_losses = {
    'cross-entropy': tf.keras.losses.CategoricalCrossentropy(from_logits=True)
}


# Available metrics. Add keys with new metrics here.
segmentation_metrics = {
    'accuracy': tf.keras.metrics.CategoricalAccuracy(name='accuracy')
}


def cropped_loss(loss_fn):
    """ Wraps loss function. Crops the labels to match the logits size. """

    def _loss_fn(labels, logits):
        logits_shape = tf.shape(logits)
        labels_crop = tf.image.resize_with_crop_or_pad(labels, logits_shape[1], logits_shape[2])

        return loss_fn(labels_crop, logits)

    return _loss_fn


class CroppedMetric(tf.keras.metrics.Metric):
    """ Wraps a metric. Crops the labels to match the logits size. """

    def __init__(self, metric):
        super().__init__(name=metric.name, dtype=metric.dtype)
        self.metric = metric

    def update_state(self, y_true, y_pred, sample_weight=None):
        logits_shape = tf.shape(y_pred)
        labels_crop = tf.image.resize_with_crop_or_pad(y_true, logits_shape[1], logits_shape[2])

        return self.metric.update_state(labels_crop, y_pred, sample_weight)

    def result(self):
        return self.metric.result()

    def reset_states(self):
        return self.metric.reset_states()

    def get_config(self):
        return self.metric.get_config()

class VisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_images, log_dir, time_index=0, rgb_indices=[0,1,2]):
        super().__init__()

        self.val_images = val_images
        self.time_index = time_index
        self.rgb_indices = rgb_indices

        self.file_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'predictions'))

    def plot_predictions(self, input_image, labels, predictions, n_classes):
        fig,(ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(18, 5))

        scaled_image = np.clip(input_image*1.5, 0., 1.)
        ax1.imshow(scaled_image)
        ax1.title.set_text('Input image (first slice)')

        cnorm = mpl.colors.NoNorm()
        cmap = plt.cm.get_cmap('Set3', n_classes)

        ax2.imshow(labels, cmap=cmap, norm=cnorm)
        ax2.title.set_text('Labeled classes')

        img = ax3.imshow(predictions, cmap=cmap, norm=cnorm)
        ax3.title.set_text('Predicted classes')

        plt.colorbar(img, ax=[ax1,ax2,ax3], shrink=0.8, ticks=list(range(n_classes)))

        return fig

    def on_epoch_end(self, epoch, logs=None):

        images = []
        for image, labels_raw in self.val_images:
            pred_raw = self.model.predict(image)
            pred_shape = tf.shape(pred_raw)

            if image.ndim == 5:
                image = image[:, self.time_index, :, :, :]

            labels_raw = tf.image.resize_with_crop_or_pad(labels_raw, pred_shape[1], pred_shape[2])
            image = tf.image.resize_with_crop_or_pad(image, pred_shape[1], pred_shape[2])

            pred = np.argmax(pred_raw, axis=-1)
            labels = np.argmax(labels_raw, axis=-1)
            num_classes = labels_raw.shape[-1]

            image = image.numpy()[...,self.rgb_indices]

            fig = self.plot_predictions(image[0], labels[0], pred[0], num_classes)
            img = plot_to_image(fig)

            images.append(img)

        images = tf.concat(images, axis=0)

        with self.file_writer.as_default():
            tf.summary.image('predictions', images, step=epoch)

class BaseSegmentationModel(BaseModel):
    """ Base for segmentation models. """

    class _Schema(Schema):
        learning_rate = fields.Float(missing=None, description='Learning rate used in training.', example=0.01)
        loss = fields.String(missing='cross-entropy', description='Loss function used for training.',
                             validate=OneOf(segmentation_losses.keys()))
        metrics = fields.List(fields.String, missing=['accuracy'], description='List of metrics used for evaluation.',
                              validate=ContainsOnly(segmentation_metrics.keys()))

    def prepare(self, optimizer=None, loss=None, metrics=None, **kwargs):
        """ Prepares the model. Optimizer, loss and metrics are read using the following protocol:
        * If an argument is None, the default value is used from the configuration of the model.
        * If an argument is a key contained in segmentation specific losses/metrics, those are used.
        * Otherwise the argument is passed to `compile` as is.

        """

        # Read defaults if None
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)

        if loss is None:
            loss = self.config.loss

        if metrics is None:
            metrics = self.config.metrics

        # Wrap loss function
        if loss in segmentation_losses:
            loss = segmentation_losses[loss]
        wrapped_loss = cropped_loss(loss)

        # Wrap metrics
        wrapped_metrics = []
        for metric in metrics:
            if metric in segmentation_metrics:
                metric = segmentation_metrics[metric]

            wrapped_metric = CroppedMetric(metric)
            wrapped_metrics.append(wrapped_metric)

        self.compile(optimizer=optimizer, loss=wrapped_loss, metrics=wrapped_metrics, **kwargs)
