import tensorflow as tf
import os


class BasePredict:
    def __init__(self, config, data, logger, graph_name="prefix"):
        self.config = config
        self.data = data
        self.logger = logger
        self.predicted_proba = None
        self.predicted_labels = None
        self.score = None
        self.graph_name = graph_name
        self.graph = self.loader()
        self.pred_dir = self.config.exp_dir + os.sep + self.config.exp_name + os.sep + self.config.pred_dir
        if not os.path.exists(self.pred_dir):
            os.mkdir(self.pred_dir)
        self.counter = 0

    def loader(self):
        """ Load existing proto-buffer model

            :return: Tensorflow graph for deep learning model
        """
        # Load the protobuf file from the disk
        with tf.gfile.GFile(self.config.checkpoint_dir + os.sep + self.config.model_name, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        # Import the graph_def into a new Graph
        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            tf.import_graph_def(graph_def, name=self.graph_name)
        return graph

    def predict(self, predict_proba=True):
        """ Saves to eopatch predicted labels, probabilities per class and plots accuracy score """
        raise NotImplementedError
