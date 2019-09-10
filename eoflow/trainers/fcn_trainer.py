from eoflow.base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class FCNTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(FCNTrainer, self).__init__(sess, model, data, config, logger)
        # mini-batch counter
        self.counter = 0

    def train_epoch(self):
        logging.info("Running epoch {}".format(self.sess.run(self.model.cur_epoch_tensor)))
        # initialise bar display
        loop = tqdm(range(self.config.num_iter_per_epoch))
        acc_epoch, loss_epoch = [], []
        # run epoch
        for _ in loop:
            # run mini-batch optimisation
            loss, acc = self.train_step()
            acc_epoch.append(acc)
            loss_epoch.append(loss)
            # log summaries for each mini-batch
            self.logger.summarize(self.counter, summaries_dict={'loss': loss,
                                                                'acc': acc})
            # run evaluation on cross-validation data set every cval_iterations
            if not np.mod(self.counter, self.config.cval_iterations):
                cval_loss, cval_acc = self.cval_step()
                # log summaries
                self.logger.summarize(self.counter, summarizer="cval", summaries_dict={'loss': cval_loss,
                                                                                       'acc': cval_acc})
            # update mini-batch counter
            self.counter += 1
        logging.info("Median accuracy for epoch {0} is {1}".format(self.sess.run(self.model.cur_epoch_tensor),
                                                                   np.median(np.array(acc_epoch))))
        logging.info("Median loss for epoch {0} is {1}".format(self.sess.run(self.model.cur_epoch_tensor),
                                                                   np.median(np.array(loss_epoch))))
        del acc_epoch, loss_epoch
        # save model
        self.model.save(self.sess)

    def train_step(self):
        # retrieve batch train data
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        if batch_x is not None and batch_y is not None:
            # build feed dictionary
            feed_dict = {self.model.x: batch_x,
                         self.model.y: batch_y,
                         self.model.is_training: True,
                         self.model.keep_prob: self.config.keep_prob}
            # run batch optimisation
            _, loss, acc = self.sess.run([self.model.train_step,
                                          self.model.loss,
                                          self.model.accuracy],
                                         feed_dict=feed_dict)
        else:
            acc, loss = np.nan, np.nan
        return loss, acc

    def cval_step(self):
        # retrieve batch cval data
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size, is_training=False))
        if batch_x is not None and batch_y is not None:
            # build feed dictionary
            feed_dict = {self.model.x: batch_x,
                         self.model.y: batch_y,
                         self.model.is_training: False,
                         self.model.keep_prob: 1}
            # run batch optimisation
            _, loss, acc = self.sess.run([self.model.train_step,
                                          self.model.loss,
                                          self.model.accuracy],
                                         feed_dict=feed_dict)
        else:
            acc, loss = np.nan, np.nan
        return loss, acc
