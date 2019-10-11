import os

import tensorflow as tf
from marshmallow import Schema, fields

from ..base import Configurable, BaseTask, BaseInput, ModelMode
from ..base.configuration import ObjectConfiguration
from ..utils import parse_classname, create_dirs

class TrainTask(BaseTask):
    class TrainTaskConfig(Schema):
        num_epochs = fields.Int(required=True, description='Number of epochs used in training', example=50)
        output_directory = fields.String(required=True, description='Directory of the model output', example='/tmp/model/')

        input_config = fields.Nested(nested=ObjectConfiguration, required=True, description="Input type and configuration.")

        save_steps = fields.Int(missing=100, description="Number of training steps between model checkpoints.")
        summary_steps = fields.Int(missing=10, description="Number of training steps between recording summaries.")
        progress_steps = fields.Int(missing=100, description="Number of training steps between writing progress messages.")

    def parse_input(self):
        input_config = self.config.input_config
        classname, config = input_config.classname, input_config.config

        cls = parse_classname(classname)
        if not issubclass(cls, BaseInput):
            raise ValueError("Data input class does not inherit from BaseInput.")

        model_input = cls(config)

        dataset_fn = lambda: model_input.get_dataset()
        return dataset_fn

    def run(self):
        dataset_fn = self.parse_input()
        
        self.model.train(dataset_fn,
                         num_epochs=self.config.num_epochs,
                         output_directory=self.config.output_directory,
                         save_steps=self.config.save_steps,
                         summary_steps=self.config.summary_steps,
                         progress_steps=self.config.progress_steps
                         )
