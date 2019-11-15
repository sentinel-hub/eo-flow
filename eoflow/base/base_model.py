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
