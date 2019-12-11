import argparse
import json

from marshmallow import Schema, fields

from .base import BaseModel, BaseTask
from .base.configuration import ObjectConfiguration, Config
from .utils import parse_classname


class ExecutionConfig(Schema):
    model = fields.Nested(ObjectConfiguration, required=True, description='Model configuration')
    task = fields.Nested(ObjectConfiguration, required=True, description='Task configuration')


def execute(config_file):
    """Executes a workflow defined in a config file."""

    with open(config_file) as file:
        config = json.load(file)

    config = Config(ExecutionConfig().load(config))

    # Parse model config
    model_cls = parse_classname(config.model.classname)
    if not issubclass(model_cls, BaseModel):
        raise ValueError("Model class does not inherit from BaseModel.")
    model = model_cls(config.model.config)

    # Parse task config
    task_cls = parse_classname(config.task.classname)
    if not issubclass(task_cls, BaseTask):
        raise ValueError("Task class does not inherit from BaseTask.")
    task = task_cls(model, config.task.config)

    # Run task
    task.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Executes a workflow described in a provided config file.')

    parser.add_argument('config_file', type=str, help='Path to the configuration file.')

    args = parser.parse_args()

    execute(config_file=args.config_file)
