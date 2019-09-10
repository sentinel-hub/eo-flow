from eoflow.base.base_predict import BasePredict
from eolearn.core import EOPatch, FeatureType
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import itertools
import os


class TFCNPredict(BasePredict):
    def __init__(self, config, data, logger):
        super(TFCNPredict, self).__init__(config, data, logger)

    def _get_sampling_indices(self, height, width):
        """ Compute indices of padded input image where patches are extracted

        :param height: Height of input image to be classified
        :type height: int
        :param width: Width of input image to be classified
        :type width: int
        :return: sample indices along the height, sample indices along the width
        :rtype: list of tuples, list of tuples
        """
        hh, ww = np.arange(height), np.arange(width)
        h_samples = [(h_min, h_max) for h_min, h_max in
                     zip(hh[::self.config.im_size[1] - self.config.spatial_cropping - self.config.prediction_overlap],
                         hh[self.config.im_size[1]::self.config.im_size[1] - self.config.spatial_cropping -
                            self.config.prediction_overlap])]
        h_tail = (height - self.config.im_size[1], height)
        if h_samples[-1] != h_tail:
            h_samples.append(h_tail)

        w_samples = [(w_min, w_max) for w_min, w_max in
                     zip(ww[::self.config.im_size[2] - self.config.spatial_cropping - self.config.prediction_overlap],
                         ww[self.config.im_size[2]::self.config.im_size[2] - self.config.spatial_cropping -
                            self.config.prediction_overlap])]
        w_tail = (width - self.config.im_size[2], width)
        if w_samples[-1] != w_tail:
            w_samples.append(w_tail)

        # if len(w_samples) != len(h_samples):
        #     raise ValueError("Different number of sampling patches")

        return h_samples, w_samples

    def _get_normalisation_mask(self, height, width, height_indices, width_indices):
        """ Compute mask where output patches overlap

        :param height: Height of input image to be classified
        :type height: int
        :param width: Width of input image to be classified
        :type width: int
        :param height_indices: List of patch indices along the image height
        :type height_indices: list of tuples
        :param width_indices: List of patch indices along the image width
        :type width_indices: list of tuples
        :return: Normalisation mask
        :rtype: numpy array
        """
        mask = np.zeros((height + self.config.spatial_cropping + 2*self.config.prediction_overlap,
                         width + self.config.spatial_cropping + 2*self.config.prediction_overlap,
                         self.config.n_classes), dtype=np.uint8)

        for hi, wi in itertools.product(height_indices, width_indices):
            mask[hi[0] + self.config.spatial_cropping // 2:hi[1] - self.config.spatial_cropping // 2,
                 wi[0] + self.config.spatial_cropping // 2:wi[1] - self.config.spatial_cropping // 2, :] += \
                    np.ones((self.config.lb_size[0], self.config.lb_size[1], self.config.n_classes), np.uint8)

        mask[mask == 0] = 1

        return mask

    def _merge_and_normalise(self, logits, height, width, height_indices, width_indices, norm_mask):
        """ Merge the output patches back into one image

        :param logits: Array storing the output of the U-net (output of softmax). Shape of array is
                        N_PATCHESxOHxOWxN_CLASSES
        :type logits: numpy array
        :param height: Height of input image to be classified
        :type height: int
        :param width: Width of input image to be classified
        :type width: int
        :param height_indices: Indices defining the location of the patches to be extracted along the height of the
                                input image
        :type height_indices: list of tuples
        :param width_indices: Indices defining the location of the patches to be extracted along the width of the
                                input image
        :type width_indices: list of tuples
        :param norm_mask: Mask counting the overlap of hte extracted patches. It's used to normalise the combined prediction
        :type norm_mask: numpy array
        :return: Predicted mask, argmax along the class dimension
        :rtype: numpy array
        """
        pred_mask = np.zeros((height + self.config.spatial_cropping + 2*self.config.prediction_overlap,
                              width + self.config.spatial_cropping + 2*self.config.prediction_overlap,
                              self.config.n_classes), dtype=np.float32)

        batch = 0
        for hi, wi in itertools.product(height_indices, width_indices):
            pred_mask[hi[0] + self.config.spatial_cropping // 2:hi[1] - self.config.spatial_cropping // 2,
                      wi[0] + self.config.spatial_cropping // 2:wi[1] - self.config.spatial_cropping // 2,
                      :] += logits[batch]
            batch += 1
        # Normalise prediction (average of overlapping values)
        pred_mask /= norm_mask
        # Find maximum of combined softmax (pseud-probability)
        return pred_mask

    def predict(self, predict_proba=True, log_accuracy=True):
        loop = tqdm(range(len(self.data.cval_set)))
        for _ in loop:
            image, labels = next(self.data.next_batch(1, is_training=False, random=False))
            eopatch_filename, = self.data.get_batch_filenames()
            self.predict_step(image, labels, eopatch_filename, predict_proba=predict_proba, log_accuracy=log_accuracy)
            self.counter += 1
        return

    def predict_step(self, image, labels, filename, predict_proba=True, log_accuracy=True):
        """ Predict step

        """
        # Load graph and get placeholders and tensors
        images = self.graph.get_tensor_by_name('{:s}/images:0'.format(self.graph_name))
        keep_prob = self.graph.get_tensor_by_name('{:s}/keep_prob:0'.format(self.graph_name))
        softmax = self.graph.get_tensor_by_name('{:s}/{:s}:0'.format(self.graph_name, self.config.node_names))

        batch_size, n_frames, height, width, depth = image.shape

        # Check dimensions agree to specification
        if self.config.im_size[-1] != depth or self.config.im_size[0] != n_frames:
            raise ValueError("Wrong number of frames/bands provided as input to model")

        if height < self.config.im_size[1] or width < self.config.im_size[2]:
            raise ValueError("Input tile is smaller than input patch. Increase spatial resolution")

        # Compute sampling indices to cover the entire input image
        h_samples, w_samples = self._get_sampling_indices(height + self.config.spatial_cropping + 2*self.config.prediction_overlap,
                                                          width + self.config.spatial_cropping + 2*self.config.prediction_overlap)
        n_patches = len(h_samples) * len(w_samples)
        # Clip image to [0,1] range and pad it using reflection
        padding = ((0, 0),
                   (0, 0),
                   (self.config.spatial_cropping // 2 + self.config.prediction_overlap,
                    self.config.spatial_cropping // 2 + self.config.prediction_overlap),
                   (self.config.spatial_cropping // 2 + self.config.prediction_overlap,
                    self.config.spatial_cropping // 2 + self.config.prediction_overlap),
                   (0, 0))

        image = np.clip(image, self.config.clip_low, self.config.clip_high)
        image = np.pad(image, padding, mode='reflect')

        # Break input image into patches of suitable input size for U-net architecture
        img_batch = [image[:, :, hi[0]:hi[1], wi[0]:wi[1], :]
                     for hi, wi in itertools.product(h_samples, w_samples)]
        img_batch = np.concatenate(img_batch, axis=0)
        del image

        # Array holding the output of the U-net
        logits = np.zeros(((n_patches,) + tuple(self.config.lb_size) + (self.config.n_classes,)), dtype=np.float32)

        # Run prediction in mini-batches
        with tf.Session(graph=self.graph) as sess:

            mini_batches = np.split(np.arange(n_patches),
                                    np.arange(n_patches)[self.config.batch_size::self.config.batch_size])

            for mb in mini_batches:
                tmp_logits = sess.run(softmax,
                                      feed_dict={images: img_batch[mb],
                                                 keep_prob: 1})
                logits[mb] = tmp_logits.reshape((len(mb),) + tuple(self.config.lb_size) + (self.config.n_classes,))
                del tmp_logits

        del softmax, images, keep_prob, img_batch

        # This normalisation mask handles overlaps between patches
        normalisation_mask = self._get_normalisation_mask(height, width, h_samples, w_samples)

        # Merge the patches back into an image of dimensions as input image
        self.predicted_proba = self._merge_and_normalise(logits, height, width, h_samples, w_samples, normalisation_mask)

        # Crop probabilities
        self.predicted_proba = self.predicted_proba[
                               self.config.spatial_cropping // 2 + self.config.prediction_overlap:self.config.spatial_cropping // 2 + self.config.prediction_overlap + height,
                               self.config.spatial_cropping // 2 + self.config.prediction_overlap:self.config.spatial_cropping // 2 + self.config.prediction_overlap + width, :]
        self.predicted_labels = np.argmax(self.predicted_proba, axis=-1).astype(np.uint8)

        if labels is not None and log_accuracy:
            accuracy = np.mean(np.equal(self.predicted_labels, np.argmax(labels, axis=-1).squeeze()).astype(np.float32))
            self.logger.summarize(self.counter, summarizer="test", summaries_dict={'loss': np.array(0.0),
                                                                                   'acc': np.array(accuracy)})

        # Save to new EOPatch predicted probs and labels
        _, eopatch_name = os.path.split(filename)

        eopatch = EOPatch.load(filename, features=[(FeatureType.BBOX, FeatureType.META_INFO)])
        if predict_proba:
            eopatch.data_timeless['PREDICTED_PROBA'] = self.predicted_proba
        eopatch.mask_timeless['PREDICTED_LABELS'] = self.predicted_labels[..., np.newaxis]

        eopatch.save(self.pred_dir + os.sep + eopatch_name)

        return


