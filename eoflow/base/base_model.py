import tensorflow as tf
import logging
import os
from enum import Enum

from . import Configurable
from ..utils import create_dirs

class ModelMode(Enum):
    TRAIN = 1
    EVALUATE = 2
    PREDICT = 3
    EXPORT = 4

class BaseModel(Configurable):

    METRICS_NAME_SCOPE = 'metrics'

    def __init__(self, config_specs):
        super().__init__(config_specs)

        self.training_summaries = []
        self.validation_summaries = []
        self.validation_update_ops = []

    # Initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def clear_graph(self):
        tf.reset_default_graph()
        self.training_summaries = []
        self.validation_summaries = []
        self.validation_update_ops = []

    def add_training_summary(self, summary):
        """Adds a summary to the list of recorded summaries."""

        self.training_summaries.append(summary)

    def add_validation_metric(self, metric_fn, name):
        """Adds a summary to the list of recorded summaries."""

        with tf.name_scope(self.METRICS_NAME_SCOPE):
            metric_val, metric_update_op = metric_fn()

        self.validation_update_ops.append(metric_update_op)

        summary = tf.summary.scalar(name, metric_val)
        self.validation_summaries.append(summary)

    def get_merged_training_summaries(self):
        """Merges all the specified summaries and returns the merged summary tensor."""

        if len(self.training_summaries) > 0:
            return tf.summary.merge(self.training_summaries)
        else:
            return tf.constant("")

    def get_validation_update_op(self):
        return tf.group(*self.validation_update_ops)

    def get_validation_init_op(self):
        variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=self.METRICS_NAME_SCOPE)
        init_op = tf.variables_initializer(variables, 'validation_init')

        return init_op

    def get_merged_validation_summaries(self):
        """Merges all the specified summaries and returns the merged summary tensor."""

        if len(self.validation_summaries) > 0:
            return tf.summary.merge(self.validation_summaries)
        else:
            return tf.constant("")

    def build_model(self, features, labels, mode):
        """Builds the model for the provided input features and labels.

        :param features: Input features tensor. Can be a single tensor or a dict of tensors.
        :type features: tf.tensor | dict(str, tf.tensor)
        :param labels: Labels tensor. Can be a single tensor or a dict of tensors.
        :type labels: tf.tensor | dict(str, tf.tensor)
        :param mode: Mode to use for building the model
        :type mode: eoflow.base.ModelMode
        """
        raise NotImplementedError

    def train(self, dataset_fn, num_epochs, output_directory, save_steps=100, summary_steps=10, progress_steps=10):
        """ Trains the model on a given dataset. Takes care of saving the model and recording summaries.

        :param dataset_fn: A function that builds and returns a tf.data.Dataset containing the input training data.
            The dataset must be of shape (features, labels) where features and labels contain the data
            in the shape required by the model.
        :type dataset_fn: function
        :param num_epochs: Number of epochs.
        :type num_epochs: int
        :param output_directory: Output directory, where the model checkpoints and summaries are saved.
        :type output_directory: str
        :param save_steps: Number of steps between saving model checkpoints.
        :type save_steps: int
        :param summary_steps: Number of steps between recodring summaries.
        :type summary_steps: int
        :param progress_steps: Number of steps between outputing progress to stdout.
        :type progress_steps: int
        """

        # Clear graph
        self.clear_graph()

        with tf.Session() as sess:
            # Build the dataset
            dataset = dataset_fn()
            iterator = dataset.make_initializable_iterator()
            features, labels = iterator.get_next()

            # Build model
            self.init_global_step()
            train_op, loss_op, summaries_op = self.build_model(features, labels, ModelMode.TRAIN)

            # Create saver
            step_tensor = self.global_step_tensor
            checkpoint_dir = os.path.join(output_directory, 'checkpoints')
            create_dirs([checkpoint_dir])
            checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
            saver = tf.train.Saver()

            # Restore latest checkpoint if it exits
            checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
            if checkpoint_file is not None:
                print("Restoring checkpoint: %s" % checkpoint_file)
                saver.restore(sess, checkpoint_file)

            # Create summary writer
            create_dirs([checkpoint_dir])
            summary_writer = tf.summary.FileWriter(output_directory, sess.graph)

            # Initialize variables
            initializer = tf.global_variables_initializer()
            sess.run(initializer)

            # Train
            try:
                training_step = 1
                for e in range(num_epochs):
                    sess.run(iterator.initializer)
                    print("Epoch %d/%d" % (e+1, num_epochs))

                    while True:
                        try:
                            # Compute and record summaries every summary_steps
                            if training_step % summary_steps == 0:
                                _, loss, step, summaries = sess.run([train_op, loss_op, step_tensor, summaries_op])

                                summary_writer.add_summary(summaries, global_step=step)
                            else:
                                _, loss, step = sess.run([train_op, loss_op, step_tensor])

                            # Show progress
                            if training_step % progress_steps == 0:
                                print("Step %d: %f" % (step, loss))

                            # Model saving
                            if training_step % save_steps == 0:
                                print("Saving checkpoint at step %d." % step)
                                saver.save(sess, checkpoint_path, global_step=step)

                            training_step += 1
                        except tf.errors.OutOfRangeError:
                            break

            # Catch user interrupt
            except KeyboardInterrupt:
                print("Training interrupted by user.")

            # Save at the end of training
            print("Saving checkpoint at step %d." % step)
            saver.save(sess, checkpoint_path, global_step=step)

    def predict(self, dataset_fn, model_directory):
        """ Runs the prediction on the model with the provided dataset
        
        :param dataset_fn: A function that builds and returns a tf.data.Dataset containing the input data.
        :type dataset_fn: function
        :param model_directory: Model directory that was used in training (`output_directory`).
        :type model_directory: str

        :return: List of predictions. Structure of predictions is defined by the model.
        :rtype: list
        """
        # Clear graph
        self.clear_graph()

        with tf.Session() as sess:
            # Build the dataset
            dataset = dataset_fn()
            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()

            # Build model
            self.init_global_step()
            predictions_op = self.build_model(features, labels, ModelMode.PREDICT)

            # Restore latest checkpoint
            checkpoint_dir = os.path.join(model_directory, 'checkpoints')
            checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
            if checkpoint_file is None:
                raise ValueError("No checkpoints found in the model directory.")

            print("Restoring checkpoint: %s" % checkpoint_file)
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_file)

            # Predict
            predictions_list = []
            while True:
                try:
                    predictions = sess.run(predictions_op)
                    predictions_list.append(predictions)
                except tf.errors.OutOfRangeError:
                    break

            return predictions_list
