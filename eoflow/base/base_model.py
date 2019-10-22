import tensorflow as tf
import logging
import os
from enum import Enum

from . import Configurable
from ..utils import create_dirs, get_common_shape

class ModelHeads():
    TRAIN = 1
    EVALUATE = 2
    PREDICT = 3

    class TrainHead:
        def __init__(self, train_op, loss_op, summaries_op):
            self.train_op = train_op
            self.loss_op = loss_op
            self.summaries_op = summaries_op
    
    class EvaluateHead:
        def __init__(self, metric_init_op, metric_update_op, metric_summaries_op, metric_values_op_dict):
            self.metric_init_op = metric_init_op
            self.metric_update_op = metric_update_op
            self.metric_summaries_op = metric_summaries_op
            self.metric_values_op_dict = metric_values_op_dict

    class PredictHead:
        def __init__(self, prediction_op):
            self.prediction_op = prediction_op

class BaseModel(Configurable):

    METRICS_NAME_SCOPE = 'metrics'

    def __init__(self, config_specs):
        super().__init__(config_specs)

        self.training_summaries = []
        self.validation_summaries = []
        self.validation_metrics = []

    # Initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def clear_graph(self):
        tf.reset_default_graph()
        self.training_summaries = []
        self.validation_summaries = []
        self.validation_metrics = []

    def add_training_summary(self, summary):
        """Adds a summary to the list of recorded summaries."""

        self.training_summaries.append(summary)

    def add_validation_metric(self, metric_fn, name):
        """Adds a summary to the list of recorded summaries."""

        with tf.name_scope(self.METRICS_NAME_SCOPE):
            metric_val, metric_update_op = metric_fn()

        # Add metric to the list
        self.validation_metrics.append((name, metric_val, metric_update_op))

        # Create summary and add it to the list
        summary = tf.summary.scalar(name, metric_val)
        self.validation_summaries.append(summary)

    def get_merged_training_summaries(self):
        """Merges all the specified summaries and returns the merged summary tensor."""

        if len(self.training_summaries) > 0:
            return tf.summary.merge(self.training_summaries)
        else:
            return tf.constant("")

    def get_merged_validation_ops(self):
        """ Prepares all the needed ops for performing validation. Init, update, summary and metric value ops. """

        # Metric initializer op (reset counters)
        variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=self.METRICS_NAME_SCOPE)
        init_op = tf.variables_initializer(variables, 'validation_init')

        # Merge update ops into a single op
        update_ops = [update_op for name, value_op, update_op in self.validation_metrics]
        merged_update_op = tf.group(*update_ops)

        # Merge summaries
        if len(self.validation_summaries) > 0:
            summary_op = tf.summary.merge(self.validation_summaries)
        else:
            summary_op = tf.constant("")

        # Dict of value ops
        value_ops = {name:value_op for name, value_op, update_op in self.validation_metrics}


        return init_op, merged_update_op, summary_op, value_ops

    def build_model(self, features, labels, is_train_tensor, model_heads):
        """Builds the model for the provided input features and labels.

        :param features: Input features tensor. Can be a single tensor or a dict of tensors.
        :type features: tf.tensor | dict(str, tf.tensor)
        :param labels: Labels tensor. Can be a single tensor or a dict of tensors.
        :type labels: tf.tensor | dict(str, tf.tensor)
        :param model_heads: List of model heads to build and return
        :type mode: list(ModelHeads)
        :param is_train_tensor: bool tensor specifying the mode of the network (training or predicting).
        :type is_train_tensor: tf.Tensor
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
            is_train_tensor = tf.constant(True)
            train_head = self.build_model(features, labels, is_train_tensor, [ModelHeads.TRAIN])[0]

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
                                _, loss, step, summaries = sess.run([train_head.train_op, train_head.loss_op, step_tensor, train_head.summaries_op])

                                summary_writer.add_summary(summaries, global_step=step)
                            else:
                                _, loss, step = sess.run([train_head.train_op, train_head.loss_op, step_tensor])

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

    def evaluate(self, model_directory, dataset_fn):
        # Clear graph
        self.clear_graph()

        with tf.Session() as sess:
            # Build the dataset
            dataset = dataset_fn()
            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()

            # Build model
            self.init_global_step()
            is_train_tensor = tf.constant(False)
            eval_head = self.build_model(features, labels, is_train_tensor, [ModelHeads.EVALUATE])[0]

            # Restore latest checkpoint
            checkpoint_dir = os.path.join(model_directory, 'checkpoints')
            checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
            if checkpoint_file is None:
                raise ValueError("No checkpoints found in the model directory.")

            print("Restoring checkpoint: %s" % checkpoint_file)
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_file)

            # Initialize metrics
            sess.run(eval_head.metric_init_op)
            while True:
                try:
                    sess.run(eval_head.metric_update_op)
                except tf.errors.OutOfRangeError:
                    break

            # Run metric value ops
            metrics = sess.run(eval_head.metric_values_op_dict)

            return metrics

    def train_and_evaluate(self, train_dataset_fn, val_dataset_fn, num_epochs, iterations_per_epoch, output_directory,
                           save_steps=100, summary_steps=10, progress_steps=10, validation_step=10):
        # Clear graph
        self.clear_graph()

        with tf.Session() as sess:
            # Build the datasets
            train_dataset = train_dataset_fn().repeat()
            val_dataset = val_dataset_fn()

            # Get common shape of the datasets (they may differ in batch size, etc.)
            shapes_train = [shape.as_list() for shape in train_dataset.output_shapes]
            shapes_val = [shape.as_list() for shape in val_dataset.output_shapes]
            
            common_shapes = tuple(get_common_shape(shape1, shape2) for shape1, shape2 in zip(shapes_train, shapes_val))

            # Dataset selector placeholder
            handle = tf.placeholder(tf.string, shape=[])

            # Switchable iterator (can switch between train and val)
            iterator = tf.data.Iterator.from_string_handle(
                        handle, train_dataset.output_types, common_shapes)
            features, labels = iterator.get_next()

            # Get dataset iterators
            train_iterator = train_dataset.make_one_shot_iterator()
            val_iterator = val_dataset.make_initializable_iterator()

            # Get handles to select which dataset iterator to use
            train_handle = sess.run(train_iterator.string_handle())
            val_handle = sess.run(val_iterator.string_handle())

            # Build model
            self.init_global_step()
            is_train = tf.placeholder(tf.bool, shape=(), name='is_train')
            train_head, eval_head = self.build_model(features, labels, is_train, [ModelHeads.TRAIN, ModelHeads.EVALUATE])

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
            train_summary_writer = tf.summary.FileWriter(os.path.join(output_directory, 'train'), sess.graph)
            val_summary_writer = tf.summary.FileWriter(os.path.join(output_directory, 'val'))

            # Initialize variables
            initializer = tf.global_variables_initializer()
            sess.run(initializer)

            # Train
            try:
                training_step = 1
                for e in range(num_epochs):
                    
                    print("Epoch %d/%d" % (e+1, num_epochs))

                    print('Training...')
                    # Train for iterations_per_epoch steps
                    for _ in range(iterations_per_epoch):
                        # Compute and record summaries every summary_steps
                        if training_step % summary_steps == 0:
                            _, loss, step, summaries = sess.run([train_head.train_op, train_head.loss_op, step_tensor, train_head.summaries_op], {handle: train_handle, is_train: True})

                            train_summary_writer.add_summary(summaries, global_step=step)
                        else:
                            _, loss, step = sess.run([train_head.train_op, train_head.loss_op, step_tensor], {handle: train_handle, is_train: True})

                        # Show progress
                        if training_step % progress_steps == 0:
                            print("Step %d: %f" % (step, loss))

                        # Model saving
                        if training_step % save_steps == 0:
                            print("Saving checkpoint at step %d." % step)
                            saver.save(sess, checkpoint_path, global_step=step)

                        training_step += 1

                    print('Evaluating...')
                    # Evaluate at the end of each epoch
                    sess.run(val_iterator.initializer)
                    sess.run(eval_head.metric_init_op)
                    while True:
                        try:
                            sess.run(eval_head.metric_update_op, {handle: val_handle, is_train: False})
                        except tf.errors.OutOfRangeError:
                            break
                    val_summaries = sess.run(eval_head.metric_summaries_op)
                    val_summary_writer.add_summary(val_summaries, global_step=step)
                    
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
            is_train_tensor = tf.constant(False)
            predictions_op = self.build_model(features, labels, is_train_tensor, [ModelHeads.PREDICT])

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
