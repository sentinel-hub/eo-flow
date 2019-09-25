import tensorflow as tf
from marshmallow import Schema, fields

from eoflow.base import Configurable, BaseTask, BaseInput, ModelMode
from eoflow.base.configuration import ObjectConfiguration
from eoflow.utils import parse_classname

class TrainTask(BaseTask):
    class TrainTaskConfig(Schema):
        num_epochs = fields.Int(required=True, description='Number of epochs used in training', example=50)
        output_directory = fields.String(required=True, description='Directory of the model output', example='/tmp/model/')

        input_config = fields.Nested(nested=ObjectConfiguration, required=True, description="Input type and configuration.")

    def parse_input(self):
        input_config = self.config.input_config
        classname, config = input_config.classname, input_config.config

        cls = parse_classname(classname)
        model_input = cls(config)

        if not issubclass(cls, BaseInput):
            raise ValueError("Data input class does not inherit from BaseInput.")

        dataset = model_input.get_dataset()
        return dataset

    def run(self):
        # TODO: configuration
        with tf.Session() as sess: 
            # Parse model input
            dataset = self.parse_input()

            iterator = dataset.make_initializable_iterator()
            features, labels = iterator.get_next()

            # Build model
            train_op, loss, summaries = self.model.build_model(features, labels, ModelMode.TRAIN)

            # Initialize variables
            initializer = tf.global_variables_initializer()
            sess.run(initializer)

            # Train
            for e in range(self.config.num_epochs):
                sess.run(iterator.initializer)

                while True:
                    try:
                        _, loss_value = sess.run([train_op, loss])

                        # TODO: Save summaries, save checkpoint, show progress, ...
                        # _, loss_value, summaries = sess.run([train_op, loss, summaries])

                    except tf.errors.OutOfRangeError:
                        break
