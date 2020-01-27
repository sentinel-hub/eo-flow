from marshmallow import Schema, fields

from ..base import BaseTask
from ..base.configuration import ObjectConfiguration


class TrainTask(BaseTask):
    class TrainTaskConfig(Schema):
        num_epochs = fields.Int(required=True, description='Number of epochs used in training', example=50)
        iterations_per_epoch = fields.Int(required=True, description='Number of training steps per epoch', example=100)
        model_directory = fields.String(required=True, description='Directory of the model output', example='/tmp/model/')

        input_config = fields.Nested(nested=ObjectConfiguration, required=True, description="Input type and configuration.")

        save_steps = fields.Int(missing=100, description="Number of training steps between model checkpoints.")
        summary_steps = fields.Int(missing='epoch', description="Number of training steps between recording summaries.")

    def run(self):
        dataset = self.parse_input(self.config.input_config)

        self.model.prepare()

        self.model.train(
            dataset,
            num_epochs=self.config.num_epochs,
            iterations_per_epoch=self.config.iterations_per_epoch,
            model_directory=self.config.model_directory,
            save_steps=self.config.save_steps,
            summary_steps=self.config.summary_steps
        )


class TrainAndEvaluateTask(BaseTask):
    class TrainAndEvaluateTask(Schema):
        num_epochs = fields.Int(required=True, description='Number of epochs used in training', example=50)
        iterations_per_epoch = fields.Int(required=True, description='Number of training steps per epoch', example=100)
        model_directory = fields.String(required=True, description='Directory of the model output',
                                        example='/tmp/model/')

        train_input_config = fields.Nested(nested=ObjectConfiguration, required=True,
                                           description="Input type and configuration for training.")
        val_input_config = fields.Nested(nested=ObjectConfiguration, required=True,
                                         description="Input type and configuration for validation.")

        save_steps = fields.Int(missing=100, description="Number of training steps between model checkpoints.")
        summary_steps = fields.Int(missing='epoch', description="Number of training steps between recording summaries.")

    def run(self):
        train_dataset = self.parse_input(self.config.train_input_config)
        val_dataset = self.parse_input(self.config.val_input_config)

        self.model.prepare()

        self.model.train_and_evaluate(
            train_dataset, val_dataset,
            num_epochs=self.config.num_epochs,
            iterations_per_epoch=self.config.iterations_per_epoch,
            model_directory=self.config.model_directory,
            save_steps=self.config.save_steps,
            summary_steps=self.config.summary_steps
        )
