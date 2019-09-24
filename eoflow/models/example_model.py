from ..base import BaseModel
import tensorflow as tf

from marshmallow import Schema, fields


class ExampleModel(BaseModel):
    class ExampleModelSchema(Schema):
        input_size = fields.Int(required=True, description='Input size of the model', example=784)
        output_size = fields.Int(required=True, description='Output size of the model', example=10)
        hidden_units = fields.Int(missing=512, description='Number of hidden units', example=512)

    def __init__(self, config_specs):
        super().__init__(config_specs)

        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None, self.config.input_size])
        self.y = tf.placeholder(tf.float32, shape=[None, self.config.output_size])

        # network architecture
        d1 = tf.layers.dense(self.x, self.config.hidden_units, activation=tf.nn.relu, name="dense1")
        d2 = tf.layers.dense(d1, self.config.output_size, name="dense2")

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d2))
            # self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss,
            self.train_step = tf.train.AdamOptimizer(0.01).minimize(self.loss,
                                                                                         global_step=
                                                                                         self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        # TODO: config interleave
        pass

