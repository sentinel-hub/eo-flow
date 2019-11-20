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

    def run(self):
        dataset = self.parse_input(self.config.input_config)

        # TODO: self.config.model_directory
        
        values = self.model.evaluate(dataset)
        names = self.model.metrics_names

        metrics = {name:value for name,value in zip(names, values)}

        # Display metrics
        print("Evaluation results:")
        for metric_name in metrics:
            print("{}: {}".format(metric_name, metrics[metric_name]))
