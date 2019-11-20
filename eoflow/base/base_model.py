import logging
import os
from enum import Enum

from tqdm.auto import tqdm
import tensorflow as tf

from . import Configurable
from ..utils import create_dirs, get_common_shape

class BaseModel(tf.keras.Model, Configurable):
    def __init__(self, config_specs):
        tf.keras.Model.__init__(self)
        Configurable.__init__(self, config_specs)

        self.init_model()

    def init_model(self):
        """ Called on __init__. Keras model initialization. If model does not require the inputs shape, create it here. """
        pass

    def build(self, inputs_shape):
        """ Keras method. Called once to build the model. Build the model here if the input shape is required. """
        pass

    def call(self, inputs, training=False):
        """ Runs the model with inputs. """
        pass

    def prepare(self):
        """ Prepares the model for training and evaluation. Call the compile method from here. """
        raise NotImplementedError

    def load_latest(self, model_directory):
        """ Loads weights from the latest checkpoint in the model directory. """

        checkpoints_path = os.path.join(model_directory, 'checkpoints', 'model.ckpt')

        return self.load_weights(checkpoints_path)

    def train(self,
              dataset,
              num_epochs,
              model_directory,
              save_steps='epoch',
              summary_steps=1,
              **kwargs):

        logs_path = os.path.join(model_directory, 'logs')
        checkpoints_path = os.path.join(model_directory, 'checkpoints', 'model.ckpt')

        # Tensorboard callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path, update_freq=summary_steps)

        # Checkpoint saving callback
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoints_path, save_freq=save_steps)

        return self.fit(dataset,
                        epochs=num_epochs,
                        callbacks=[tensorboard_callback, checkpoint_callback],
                        **kwargs)

    def train_and_evaluate(self, train_dataset, val_dataset, num_epochs, iterations_per_epoch, model_directory,
                           save_steps=100, summary_steps=10, **kwargs):
        """ Trains the model on a given dataset. At the end of each epoch an evaluation is performed on the provided
            validation dataset. Takes care of saving the model and recording summaries.

        :param train_dataset_fn: A function that builds and returns a tf.data.Dataset containing the input training data.
            The dataset must be of shape (features, labels) where features and labels contain the data
            in the shape required by the model.
        :type train_dataset_fn: function
        :param val_dataset_fn: Same as for `train_dataset_fn`, but for the validation data.
        :type val_dataset_fn: function
        :param num_epochs: Number of epochs.
        :type num_epochs: int
        :param iterations_per_epoch: Number of training steps to make every epoch.
            Training dataset is repeated automatically when the end is reached.
        :type iterations_per_epoch: int
        :param model_directory: Output directory, where the model checkpoints and summaries are saved.
        :type model_directory: str
        :param save_steps: Number of steps between saving model checkpoints.
        :type save_steps: int
        :param summary_steps: Number of steps between recodring summaries.
        :type summary_steps: int
        :param progress_steps: Number of steps between outputing progress to stdout.
        :type progress_steps: int
        """

        logs_path = os.path.join(model_directory, 'logs')
        checkpoints_path = os.path.join(model_directory, 'checkpoints', 'model.ckpt')

        # Tensorboard callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path, update_freq=summary_steps)

        # Checkpoint saving callback
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoints_path, save_freq=save_steps)

        # Repeat training dataset indefenetly
        train_dataset = train_dataset.repeat()

        return self.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=num_epochs,
                        steps_per_epoch=iterations_per_epoch,
                        callbacks=[tensorboard_callback, checkpoint_callback],
                        **kwargs)
