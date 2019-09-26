from ..base import BaseModel, ModelMode
import tensorflow as tf

from marshmallow import Schema, fields


class ExampleModel(BaseModel):
    class ExampleModelSchema(Schema):
        output_size = fields.Int(required=True, description='Output size of the model', example=10)
        hidden_units = fields.Int(missing=512, description='Number of hidden units', example=512)
        learning_rate = fields.Float(required=True, description='Learning rate used', example=0.01)

    def build_model(self, features, labels, mode):
        x = features
        
        d1 = tf.layers.dense(x, self.config.hidden_units, activation=tf.nn.relu, name="dense1")
        d2 = tf.layers.dense(d1, self.config.output_size, name="dense2")

        if mode == ModelMode.TRAIN:
            with tf.name_scope("loss"):
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=d2))
                optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
                train_op = optimizer.minimize(loss, global_step=self.global_step_tensor)
                
                return train_op, loss, self.get_merged_summaries()

        elif mode == ModelMode.PREDICT:
            probabilities = tf.softmax(d2)
            predictions = tf.argmax(probabilities, axis=1)

            predictions = {
                'prediction': predictions,
                'probabilities': probabilities
            }

            return predictions
        
        else:
            raise NotImplementedError

