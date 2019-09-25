import argparse
import json

from marshmallow import Schema, fields

from eoflow.base.configuration import ObjectConfiguration, Config
from eoflow.utils import parse_classname

class ExecutionConfig(Schema):
    model = fields.Nested(ObjectConfiguration, required=True, description='Model configuration')
    task = fields.Nested(ObjectConfiguration, required=True, description='Task configuration')

def execute(config_file):
    with open(config_file) as file:
        config = json.load(file)

    config = Config(ExecutionConfig().load(config))
    
    model_cls = parse_classname(config.model.classname)
    model = model_cls(config.model.config)

    task_cls = parse_classname(config.task.classname)
    task = task_cls(model, config.task.config)

    task.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Executes a workflow described in a provided config file.')

    parser.add_argument('config_file', type=str, help='Path to the configuration file.')

    args = parser.parse_args()

    execute(config_file=args.config_file)