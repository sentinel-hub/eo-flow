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
        pass

    def build(self, inputs_shape):
        pass

    def call(self, inputs, training=False):
        pass

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
