import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..utils.tf_utils import plot_to_image


class VisualizationCallback(tf.keras.callbacks.Callback):
    """ Keras Callback for saving prediction visualizations to TensorBoard. """

    def __init__(self, val_images, log_dir, time_index=0, rgb_indices=[2, 1, 0]):
        """
        :param val_images: Images to run predictions on. Tuple of (images, labels).
        :type val_images: (np.array, np.array)
        :param log_dir: Directory where the TensorBoard logs are written.
        :type log_dir: str
        :param time_index: Time index to use, when multiple time slices are available, defaults to 0
        :type time_index: int, optional
        :param rgb_indices: Indices for R, G and B bands in the input image, defaults to [0,1,2]
        :type rgb_indices: list, optional
        """
        super().__init__()

        self.val_images = val_images
        self.time_index = time_index
        self.rgb_indices = rgb_indices

        self.file_writer = tf.summary.create_file_writer(log_dir)

    @staticmethod
    def plot_predictions(input_image, labels, predictions, n_classes):
        # TODO: fix figsize (too wide?)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        scaled_image = np.clip(input_image*2.5, 0., 1.)
        ax1.imshow(scaled_image)
        ax1.title.set_text('Input image')

        cnorm = mpl.colors.NoNorm()
        cmap = plt.cm.get_cmap('Set3', n_classes)

        ax2.imshow(labels, cmap=cmap, norm=cnorm)
        ax2.title.set_text('Labeled classes')

        img = ax3.imshow(predictions, cmap=cmap, norm=cnorm)
        ax3.title.set_text('Predicted classes')

        plt.colorbar(img, ax=[ax1, ax2, ax3], shrink=0.8, ticks=list(range(n_classes)))

        return fig

    def prediction_summaries(self, step):
        images, labels = self.val_images
        preds_raw = self.model.predict(images)

        pred_shape = tf.shape(preds_raw)

        # If temporal data only use time_index slice
        if images.ndim == 5:
            images = images[:, self.time_index, :, :, :]

        # Crop images and labels to output size
        labels = tf.image.resize_with_crop_or_pad(labels, pred_shape[1], pred_shape[2])
        images = tf.image.resize_with_crop_or_pad(images, pred_shape[1], pred_shape[2])

        # Take RGB values
        images = images.numpy()[..., self.rgb_indices]

        num_classes = labels.shape[-1]

        # Get class ids
        preds_raw = np.argmax(preds_raw, axis=-1)
        labels = np.argmax(labels, axis=-1)

        vis_images = []
        for image_i, labels_i, pred_i in zip(images, labels, preds_raw):
            # Plot predictions and convert to image
            fig = self.plot_predictions(image_i, labels_i, pred_i, num_classes)
            img = plot_to_image(fig)

            vis_images.append(img)

        n_images = len(vis_images)
        vis_images = tf.concat(vis_images, axis=0)

        with self.file_writer.as_default():
            tf.summary.image('predictions', vis_images, step=step, max_outputs=n_images)

    def on_epoch_end(self, epoch, logs=None):
        self.prediction_summaries(epoch)
