import os

import tensorflow as tf

from . import Configurable


class BaseModel(tf.keras.Model, Configurable):
    def __init__(self, config_specs):
        tf.keras.Model.__init__(self)
        Configurable.__init__(self, config_specs)

        self.net = None

        self.init_model()

    def init_model(self):
        """ Called on __init__. Keras model initialization. Create model here if does not require the inputs shape """
        pass

    def build(self, inputs_shape):
        """ Keras method. Called once to build the model. Build the model here if the input shape is required. """
        pass

    def call(self, inputs, training=False):
        """ Runs the model with inputs. """
        pass

    def prepare(self, optimizer=None, loss=None, metrics=None, **kwargs):
        """ Prepares the model for training and evaluation. This method should create the
        optimizer, loss and metric functions and call the compile method of the model. The model
        should provide the defaults for the optimizer, loss and metrics, which can be overriden
        with custom arguments. """

        raise NotImplementedError

    def load_latest(self, model_directory):
        """ Loads weights from the latest checkpoint in the model directory. """

        checkpoints_path = os.path.join(model_directory, 'checkpoints', 'model.ckpt')

        return self.load_weights(checkpoints_path).expect_partial()

    def _fit(self,
             dataset,
             num_epochs,
             model_directory,
             iterations_per_epoch,
             val_dataset=None,
             save_steps=100,
             summary_steps='epoch',
             callbacks=[],
             **kwargs):
        """ Trains and evaluates the model on a given dataset, saving the model and recording summaries. """
        logs_path = os.path.join(model_directory, 'logs')
        checkpoints_path = os.path.join(model_directory, 'checkpoints', 'model.ckpt')

        # Tensorboard callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path,
                                                              update_freq=summary_steps,
                                                              profile_batch=0)

        # Checkpoint saving callback
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoints_path,
                                                                 save_best_only=True,
                                                                 save_freq=save_steps,
                                                                 save_weights_only=True)
        return self.fit(dataset,
                        validation_data=val_dataset,
                        epochs=num_epochs,
                        steps_per_epoch=iterations_per_epoch,
                        callbacks=[tensorboard_callback, checkpoint_callback] + callbacks,
                        **kwargs)

    def train(self,
              dataset,
              num_epochs,
              model_directory,
              iterations_per_epoch=None,
              save_steps=100,
              summary_steps='epoch',
              callbacks=[],
              **kwargs):
        """ Trains the model on a given dataset. Takes care of saving the model and recording summaries.

        :param dataset: A tf.data Dataset containing the input training data.
            The dataset must be of shape (features, labels) where features and labels contain the data
            in the shape required by the model.
        :type dataset: tf.data.Dataset
        :param num_epochs: Number of epochs. One epoch is equal to one pass over the dataset.
        :type num_epochs: int
        :param model_directory: Output directory, where the model checkpoints and summaries are saved.
        :type model_directory: str
        :param iterations_per_epoch: Number of training steps to make every epoch.
            Training dataset is repeated automatically when the end is reached.
        :type iterations_per_epoch: int
        :param save_steps: Number of steps between saving model checkpoints.
        :type save_steps: int
        :param summary_steps: Number of steps between recording summaries.
        :type summary_steps: str or int
        :param callbacks: Customised Keras callbacks to use during training/evaluation
        :type callbacks: tf.keras.callbacks

        Other keyword parameters are passed to the Model.fit method.
        """

        return self._fit(dataset if iterations_per_epoch is None else dataset.repeat(),
                         num_epochs=num_epochs,
                         model_directory=model_directory,
                         iterations_per_epoch=iterations_per_epoch,
                         save_steps=save_steps,
                         summary_steps=summary_steps,
                         callbacks=callbacks,
                         **kwargs)

    def train_and_evaluate(self,
                           train_dataset,
                           val_dataset,
                           num_epochs,
                           iterations_per_epoch,
                           model_directory,
                           save_steps=100, summary_steps='epoch', callbacks=[], **kwargs):
        """ Trains the model on a given dataset. At the end of each epoch an evaluation is performed on the provided
            validation dataset. Takes care of saving the model and recording summaries.

        :param train_dataset: A tf.data Dataset containing the input training data.
            The dataset must be of shape (features, labels) where features and labels contain the data
            in the shape required by the model.
        :type train_dataset: tf.data.Dataset
        :param val_dataset: Same as for `train_dataset`, but for the validation data.
        :type val_dataset: tf.data.Dataset
        :param num_epochs: Number of epochs. Epoch size is independent from the dataset size.
        :type num_epochs: int
        :param iterations_per_epoch: Number of training steps to make every epoch.
            Training dataset is repeated automatically when the end is reached.
        :type iterations_per_epoch: int
        :param model_directory: Output directory, where the model checkpoints and summaries are saved.
        :type model_directory: str
        :param save_steps: Number of steps between saving model checkpoints.
        :type save_steps: int
        :param summary_steps: Number of steps between recodring summaries.
        :type summary_steps: str or int
        :param callbacks: Customised Keras callbacks to use during training/evaluation
        :type callbacks: tf.keras.callbacks

        Other keyword parameters are passed to the Model.fit method.
        """
        return self._fit(train_dataset.repeat(),
                         num_epochs,
                         model_directory,
                         iterations_per_epoch,
                         val_dataset=val_dataset,
                         save_steps=save_steps,
                         summary_steps=summary_steps,
                         callbacks=callbacks,
                         **kwargs)
