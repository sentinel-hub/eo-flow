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
