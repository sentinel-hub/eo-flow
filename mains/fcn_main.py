import tensorflow as tf

from eoflow.data_loader import EOMonoTempDataGenerator
from eoflow.trainers import FCNTrainer
from eoflow.utils import process_config
from eoflow.models import FCNModel
from eoflow.utils import create_dirs
from eoflow.utils import get_args
from eoflow.utils import Logger

import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.log_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create an instance of the model you want
    model = FCNModel(config)
    # load model if exists
    model.load(sess)
    # create your data generator
    data = EOMonoTempDataGenerator(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = FCNTrainer(sess, model, data, config, logger)
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
