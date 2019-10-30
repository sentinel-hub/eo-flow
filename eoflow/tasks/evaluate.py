import os

import tensorflow as tf
from marshmallow import Schema, fields

from ..base import Configurable, BaseTask, BaseInput
from ..base.configuration import ObjectConfiguration
from ..utils import parse_classname, create_dirs

class EvaluateTask(BaseTask):
    class EvaluateTaskConfig(Schema):
        model_directory = fields.String(required=True, description='Directory of the model', example='/tmp/model/')

        input_config = fields.Nested(nested=ObjectConfiguration, required=True, description="Input type and configuration.")

    def parse_input(self):
        input_config = self.config.input_config
        classname, config = input_config.classname, input_config.config

        cls = parse_classname(classname)
        if not issubclass(cls, BaseInput):
            raise ValueError("Data input class does not inherit from BaseInput.")

        model_input = cls(config)

        dataset_fn = model_input.get_dataset
        return dataset_fn

    def run(self):
        dataset_fn = self.parse_input()

        metrics = self.model.evaluate(dataset_fn, self.config.model_directory)

        # Display metrics
        print("Evaluation results:")
        for metric_name in metrics:
            print("{}: {}".format(metric_name, metrics[metric_name]))
