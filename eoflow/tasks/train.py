import os

import tensorflow as tf
from marshmallow import Schema, fields

from eoflow.base import Configurable, BaseTask, BaseInput, ModelMode
from eoflow.base.configuration import ObjectConfiguration
from eoflow.utils import parse_classname, create_dirs

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
            train_op, loss_op, summaries_op = self.model.build_model(features, labels, ModelMode.TRAIN)

            # Create saver
            step_tensor = self.model.global_step_tensor
            checkpoint_dir = os.path.join(self.config.output_directory, 'checkpoints')
            create_dirs([checkpoint_dir])
            checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
            saver = tf.train.Saver()

            # Create summary writer
            create_dirs([checkpoint_dir])
            summary_writer = tf.summary.FileWriter(self.config.output_directory, sess.graph)

            # Initialize variables
            initializer = tf.global_variables_initializer()
            sess.run(initializer)

            # Train
            training_step = 1
            for e in range(self.config.num_epochs):
                sess.run(iterator.initializer)
                print("Epoch %d/%d" % (e+1, self.config.num_epochs))

                while True:
                    try:
                        # Compute and record summaries every summary_steps
                        if training_step % self.config.summary_steps == 0:
                            _, loss, step, summaries = sess.run([train_op, loss_op, step_tensor, summaries_op])

                            summary_writer.add_summary(summaries, global_step=step)
                        else:
                            _, loss, step = sess.run([train_op, loss_op, step_tensor])

                        # Show progress
                        if training_step % self.config.progress_steps == 0:
                            print("Step %d: %f" % (step, loss))

                        # Model saving
                        if training_step % self.config.save_steps == 0:
                            print("Saving checkpoint at step %d." % step)
                            saver.save(sess, checkpoint_path, global_step=step)

                        training_step += 1
                    except tf.errors.OutOfRangeError:
                        break
