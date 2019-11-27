from eoflow.base import BaseModel
import tensorflow as tf

from marshmallow import Schema, fields


class ExampleModel(BaseModel):
    """ Example implementation of a model. Builds a fully connected net with a single hidden layer. """

    class _Schema(Schema):
        output_size = fields.Int(required=True, description='Output size of the model', example=10)
        hidden_units = fields.Int(missing=512, description='Number of hidden units', example=512)
        learning_rate = fields.Float(missing=0.01, description='Learning rate for Adam optimizer', example=0.01)

    def init_model(self):
        l1 = tf.keras.layers.Dense(self.config.hidden_units, activation='relu')
        l2 = tf.keras.layers.Dense(self.config.output_size, activation='softmax')

        self.net = tf.keras.Sequential([l1, l2])

    def call(self, inputs, training=False):
        x = self.net(inputs)

        return x

    def prepare(self, optimizer=None, loss=None, metrics=None, **kwargs):
        """ Prepares the model. Optimizer, loss and metrics are read using the following protocol:
        * If an argument is None, the default value is used from the configuration of the model.
        * If an argument is a key contained in segmentation specific losses/metrics, those are used.
        * Otherwise the argument is passed to `compile` as is.

        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)

        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        metrics = tf.keras.metrics.CategoricalAccuracy(name='accuracy')

        self.compile(optimizer=optimizer, loss=loss, metrics=[metrics], **kwargs)
