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

    def __init__(self, config_specs):
        super().__init__(config_specs)

        self.summaries = []

        # init the epoch counter
        self.init_cur_epoch()

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess):
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        logging.info("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            self.saver.restore(sess, latest_checkpoint)
            logging.info("Model loaded from checkpoint {}".format(latest_checkpoint))

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def freeze_graph(self):
        """ Extract the sub graph defined by the output nodes and convert all its variables into constant

            Freezes and saves model as proto-buffer file (e.g. model.pb). The name of the output layers (if more than
            one) of the network need to be specified (e.g. "Softmax").
        """
        if not tf.gfile.Exists(self.config.checkpoint_dir):
            raise AssertionError("Export directory {:s} doesn't exists.".format(self.config.checkpoint_dir))
        if not self.config.node_names:
            raise ValueError("You need to supply the name of a node.")
        # We retrieve our checkpoint fullpath
        checkpoint = tf.train.get_checkpoint_state(self.config.checkpoint_dir)
        input_checkpoint = checkpoint.model_checkpoint_path
        # We precise the file fullname of our freezed graph
        output_graph = "/".join(input_checkpoint.split('/')[:-1]) + os.sep + self.config.model_name

        # We start a session using a temporary fresh Graph
        with tf.Session(graph=tf.Graph()) as sess:
            # We import the meta graph in the current default Graph
            saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
            # We restore the weights
            saver.restore(sess, input_checkpoint)
            # nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
            # print(nodes)
            # We use a built-in TF helper to export variables to constants
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,  # The session is used to retrieve the weights
                tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
                self.config.node_names.split(",")  # The output node names are used to select the useful nodes
            )

            # Finally we serialize and dump the output graph to the filesystem
            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())

    def add_summary(self, summary):
        """Adds a summary to the list of recorded summaries."""

        self.summaries.append(summary)

    def get_merged_summaries(self):
        """Merges all the specified summaries and returns the merged summary tensor."""

        if len(self.summaries) > 0:
            return tf.summary.merge(self.summaries)
        else:
            return tf.constant("")

    def init_saver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError

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
        # Clear graph
        tf.reset_default_graph()

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
