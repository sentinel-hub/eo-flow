import tensorflow as tf
import logging
import os

from .configurable import Configurable

class BaseModel(Configurable):

    def __init__(self, config_specs):
        super().__init__(config_specs)

        # init the global step
        self.init_global_step()
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

    def init_saver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError
