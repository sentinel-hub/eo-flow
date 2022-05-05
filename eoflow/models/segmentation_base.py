import os

import numpy as np
import tensorflow as tf
from marshmallow import Schema, fields
from marshmallow.validate import OneOf, ContainsOnly

from ..base import BaseModel

from .losses import CategoricalCrossEntropy, CategoricalFocalLoss, JaccardDistanceLoss, TanimotoDistanceLoss
from .losses import cropped_loss
from .metrics import MeanIoU, InitializableMetric, CroppedMetric, MCCMetric
from .callbacks import VisualizationCallback


# Available losses. Add keys with new losses here.
segmentation_losses = {
    'cross_entropy': CategoricalCrossEntropy,
    'focal_loss': CategoricalFocalLoss,
    'jaccard_loss': JaccardDistanceLoss,
    'tanimoto_loss': TanimotoDistanceLoss
}


# Available metrics. Add keys with new metrics here.
segmentation_metrics = {
    'accuracy': lambda: tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
    'iou': lambda: MeanIoU(default_max_classes=32),
    'precision': tf.keras.metrics.Precision,
    'recall': tf.keras.metrics.Recall,
    'mcc': MCCMetric
}


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

        class_weights = fields.Dict(missing=None, description='Dictionary mapping class id with weight. '
                                                              'If key for some labels is not specified, 1 is used.')

        prediction_visualization = fields.Bool(missing=False, description='Record prediction visualization summaries.')
        prediction_visualization_num = fields.Int(missing=5,
                                                  description='Number of images used for prediction visualization.')

    def _prepare_class_weights(self):
        """ Utility function to parse class weights """
        if self.config.class_weights is None:
            return np.ones(self.config.n_classes)
        return np.array([self.config.class_weights[iclass] if iclass in self.config.class_weights else 1.0
                         for iclass in range(self.config.n_classes)])

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

        class_weights = self._prepare_class_weights()

        # Wrap loss function
        # TODO: pass kwargs to loss from config
        if loss in segmentation_losses:
            loss = segmentation_losses[loss](from_logits=False, class_weights=class_weights)
        wrapped_loss = cropped_loss(loss)

        # Wrap metrics
        wrapped_metrics = []
        for metric in metrics:

            if metric in segmentation_metrics:
                if metric in ['precision', 'recall']:
                    wrapped_metrics += [CroppedMetric(segmentation_metrics[metric](top_k=1,
                                                                                   class_id=class_id,
                                                                                   name=f'{metric}_{class_id}'))
                                        for class_id in range(self.config.n_classes)]
                    continue
                else:
                    metric = segmentation_metrics[metric]()

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
              iterations_per_epoch=None,
              class_weights=None,
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
                           class_weights=None,
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
                                   callbacks=callbacks + custom_callbacks, **kwargs)
