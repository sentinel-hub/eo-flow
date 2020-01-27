from marshmallow import Schema, fields

from ..base import BaseTask
from ..base.configuration import ObjectConfiguration


class PredictTask(BaseTask):
    class PredictTaskConfig(Schema):
        model_directory = fields.String(required=True, description='Directory of the model', example='/tmp/model/')

        input_config = fields.Nested(nested=ObjectConfiguration, required=True,
                                     description="Input type and configuration.")

    def run(self):
        dataset_fn = self.parse_input(self.config.input_config)

        self.model.prepare() # TODO: find a way to initialize without compiling the model
        self.model.load_latest(self.config.model_directory)

        predictions_list = self.model.predict(dataset_fn)
        # TODO: something with predictions
