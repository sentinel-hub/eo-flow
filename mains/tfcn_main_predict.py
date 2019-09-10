import tensorflow as tf

from eoflow.data_loader import PredictDataGenerator
from eoflow.predictors import TFCNPredict
from eoflow.utils import process_config
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
    except RuntimeError:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.log_dir, config.checkpoint_dir])

    # read test data
    test_data = PredictDataGenerator(config)
    with tf.Session() as sess:
        test_logger = Logger(sess, config)
        # predictor class
        predictor = TFCNPredict(config, test_data, test_logger)
        predictor.predict()


if __name__ == '__main__':
    main()
