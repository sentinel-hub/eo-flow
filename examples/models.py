from eoflow.base import BaseModel, ModelHeads
import tensorflow as tf

from marshmallow import Schema, fields


class ExampleModel(BaseModel):
    """ Example implementation of a model. Builds a fully connected net with a single hidden layer. """

    # Define a schema for configuration.
    class _Schema(Schema):
        output_size = fields.Int(required=True, description='Output size of the model', example=10)
        hidden_units = fields.Int(missing=512, description='Number of hidden units', example=512)
        learning_rate = fields.Float(required=True, description='Learning rate used', example=0.01)

    def build_model(self, features, labels, is_train_tensor, model_heads):
        x = features
        
        # Build the network
        d1 = tf.layers.dense(x, self.config.hidden_units, activation=tf.nn.relu, name="dense1")
        d2 = tf.layers.dense(d1, self.config.output_size, name="dense2")

        # Build the head for training
        if ModelHeads.TRAIN in model_heads:
            # Compute loss and create a training op
            with tf.name_scope("loss"):
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=d2))
                optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
                train_op = optimizer.minimize(loss, global_step=self.global_step_tensor)

                # Add summary for loss
                self.add_training_summary(tf.summary.scalar('loss', loss))
                
                train_head = ModelHeads.TrainHead(train_op, loss, self.get_merged_training_summaries())

        # Build the head for prediction (no loss computation)
        if ModelHeads.PREDICT in model_heads:
            # Compute predictions
            probabilities = tf.nn.softmax(d2)
            predictions = tf.argmax(probabilities, axis=1)

            # Define the output of the prediction. Here a dict is used.
            predictions = {
                'prediction': predictions,
                'probabilities': probabilities
            }

            predict_head = ModelHeads.PredictHead(predictions)

        # Build the head for evaluation
        if ModelHeads.EVALUATE in model_heads:
            # Compute predictions
            probabilities = tf.nn.softmax(d2)
            predictions = tf.argmax(probabilities, axis=1)

            labels_n = tf.argmax(labels, axis=1)
            accuracy_metric_fn = lambda: tf.metrics.accuracy(labels_n, predictions)

            # Adds a metric
            self.add_validation_metric(accuracy_metric_fn, 'accuracy')

            # Automatically create init, update, summary, metric value ops for 
            # metrics provided using the `add_validation_metric` method.
            # Should fit most use cases, but allows extension with custom ops, summaries, etc.
            init_op, update_op, summary_op, value_op = self.get_merged_validation_ops()

            evaluate_head = ModelHeads.EvaluateHead(init_op, update_op, summary_op, value_op)


        # Return requested heads in a list
        heads = []
        for model_head in model_heads:
            if model_head == ModelHeads.TRAIN:
                heads.append(train_head)
            elif model_head == ModelHeads.PREDICT:
                heads.append(predict_head)
            elif model_head == ModelHeads.EVALUATE:
                heads.append(evaluate_head)
            else:
                raise NotImplementedError

        return heads
