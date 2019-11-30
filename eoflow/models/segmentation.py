import os
import logging

import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from marshmallow import Schema, fields
from marshmallow.validate import OneOf, ContainsOnly

from ..base import BaseModel
from ..utils.tf_utils import plot_to_image

from .losses import CategoricalFocalLoss, JaccardDistanceLoss
from .metrics import MeanIoU, InitializableMetric

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# Available losses. Add keys with new losses here.
segmentation_losses = {
    'cross_entropy': tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    'focal_loss': CategoricalFocalLoss(from_logits=True),
    'jaccard_loss': JaccardDistanceLoss(from_logits=True)
}


# Available metrics. Add keys with new metrics here.
segmentation_metrics = {
    'accuracy': tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
    'iou': MeanIoU(default_max_classes=32)
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
    """ Keras Callback for saving prediction visualizations to TensorBoard. """

    def __init__(self, val_images, log_dir, time_index=0, rgb_indices=[2, 1, 0]):
        """
        :param val_images: Images to run predictions on. Tuple of (images, labels).
        :type val_images: (np.array, np.array)
        :param log_dir: Directory where the TensorBoard logs are written.
        :type log_dir: str
        :param time_index: Time index to use, when multiple time slices are available, defaults to 0
        :type time_index: int, optional
        :param rgb_indices: Indices for R, G and B bands in the input image, defaults to [0,1,2]
        :type rgb_indices: list, optional
        """
        super().__init__()

        self.val_images = val_images
        self.time_index = time_index
        self.rgb_indices = rgb_indices

        self.file_writer = tf.summary.create_file_writer(log_dir)

    @staticmethod
    def plot_predictions(input_image, labels, predictions, n_classes):
        # TODO: fix figsize (too wide?)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        scaled_image = np.clip(input_image*2.5, 0., 1.)
        ax1.imshow(scaled_image)
        ax1.title.set_text('Input image')

        cnorm = mpl.colors.NoNorm()
        cmap = plt.cm.get_cmap('Set3', n_classes)

        ax2.imshow(labels, cmap=cmap, norm=cnorm)
        ax2.title.set_text('Labeled classes')

        img = ax3.imshow(predictions, cmap=cmap, norm=cnorm)
        ax3.title.set_text('Predicted classes')

        plt.colorbar(img, ax=[ax1, ax2, ax3], shrink=0.8, ticks=list(range(n_classes)))

        return fig

    def prediction_summaries(self, step):
        images, labels = self.val_images
        preds_raw = self.model.predict(images)

        pred_shape = tf.shape(preds_raw)

        # If temporal data only use time_index slice
        if images.ndim == 5:
            images = images[:, self.time_index, :, :, :]

        # Crop images and labels to output size
        labels = tf.image.resize_with_crop_or_pad(labels, pred_shape[1], pred_shape[2])
        images = tf.image.resize_with_crop_or_pad(images, pred_shape[1], pred_shape[2])

        # Take RGB values
        images = images.numpy()[..., self.rgb_indices]

        num_classes = labels.shape[-1]

        # Get class ids
        preds_raw = np.argmax(preds_raw, axis=-1)
        labels = np.argmax(labels, axis=-1)

        vis_images = []
        for image_i, labels_i, pred_i in zip(images, labels, preds_raw):
            # Plot predictions and convert to image
            fig = self.plot_predictions(image_i, labels_i, pred_i, num_classes)
            img = plot_to_image(fig)

            vis_images.append(img)

        n_images = len(vis_images)
        vis_images = tf.concat(vis_images, axis=0)

        with self.file_writer.as_default():
            tf.summary.image('predictions', vis_images, step=step, max_outputs=n_images)

    def on_epoch_end(self, epoch, logs=None):
        self.prediction_summaries(epoch)


class BaseSegmentationModel(BaseModel):
    """ Base for segmentation models. """

    class _Schema(Schema):
        n_classes = fields.Int(required=True, description='Number of classes', example=2)
        learning_rate = fields.Float(missing=None, description='Learning rate used in training.', example=0.01)
        loss = fields.String(missing='cross_entropy', description='Loss function used for training.',
                             validate=OneOf(segmentation_losses.keys()))
        metrics = fields.List(fields.String, missing=['accuracy', 'iou'],
                              description='List of metrics used for evaluation.',
                              validate=ContainsOnly(segmentation_metrics.keys()))

        prediction_visualization = fields.Bool(missing=False, description='Record prediction visualization summaries.')
        prediction_visualization_num = fields.Int(missing=5, description='Number of images used for prediction visualization.')

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

            # Initialize initializable metrics
            if isinstance(metric, InitializableMetric):
                metric.init_from_config(self.config)

            wrapped_metric = CroppedMetric(metric)
            wrapped_metrics.append(wrapped_metric)

        self.compile(optimizer=optimizer, loss=wrapped_loss, metrics=wrapped_metrics, **kwargs)

    def _get_visualization_callback(self, dataset, log_dir):
        ds = dataset.unbatch().batch(self.config.prediction_visualization_num).take(1)
        data = next(iter(ds))

        visualization_callback = VisualizationCallback(data, log_dir)
        return visualization_callback

    # Override default method to add prediction visualization
    def train(self, dataset,
              num_epochs,
              model_directory,
              iterations_per_epoch,
              callbacks=[],
              save_steps='epoch',
              summary_steps=1, **kwargs):

        custom_callbacks = []

        if self.config.prediction_visualization:
            log_dir = os.path.join(model_directory, 'logs', 'predictions')
            visualization_callback = self._get_visualization_callback(dataset, log_dir)
            custom_callbacks.append(visualization_callback)

        super().train(dataset, num_epochs, model_directory, iterations_per_epoch,
                      callbacks=callbacks + custom_callbacks, save_steps=save_steps,
                      summary_steps=summary_steps, **kwargs)

    # Override default method to add prediction visualization
    def train_and_evaluate(self,
                           train_dataset,
                           val_dataset,
                           num_epochs,
                           iterations_per_epoch,
                           model_directory,
                           validation_steps=1,
                           save_steps=100,
                           summary_steps=10,
                           callbacks=[], **kwargs):

        custom_callbacks = []

        if self.config.prediction_visualization:
            log_dir = os.path.join(model_directory, 'logs', 'predictions')
            visualization_callback = self._get_visualization_callback(val_dataset, log_dir)
            custom_callbacks.append(visualization_callback)

        super().train_and_evaluate(train_dataset, val_dataset,
                                   num_epochs, iterations_per_epoch,
                                   model_directory,
                                   save_steps=save_steps, summary_steps=summary_steps,
                                   validation_steps=validation_steps,
                                   callbacks=callbacks + custom_callbacks, **kwargs)
