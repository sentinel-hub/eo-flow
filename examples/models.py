from eoflow.base import BaseModel, ModelHeads
import tensorflow as tf

from marshmallow import Schema, fields


class ExampleModel(BaseModel):
    """ Example implementation of a model. Builds a fully connected net with a single hidden layer. """

    class _Schema(Schema):
        output_size = fields.Int(required=True, description='Output size of the model', example=10)
        hidden_units = fields.Int(missing=512, description='Number of hidden units', example=512)
    
    def init_model(self):
        l1 = tf.keras.layers.Dense(self.config.hidden_units, activation='relu')
        l2 = tf.keras.layers.Dense(self.config.output_size, activation='softmax')
        
        self.net = tf.keras.Sequential([l1,l2])
        
    def call(self, inputs, training=False):
        x = self.net(inputs)
        
        return x
