import tensorflow as tf

from eoflow.data_loader.s2_l1c_l2a_generator import OnlineEODataGenerator
from eoflow.predictors.s2_l1c_to_l2a_predictor import S2L1CToL2APredict
from eoflow.eolearn_workflows.s2_l1c_l2a import S2L1CToL2AWorkflow
from eoflow.models.s2_l1c_to_l2a_model import S2L1CToL2AModel
from eoflow.trainers.fcn_trainer import FCNTrainer
from eoflow.utils.config import process_config
from eoflow.utils.dirs import create_dirs
from eoflow.utils.utils import get_args
from eoflow.utils.logger import Logger

import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    config = None
    try:
        args = get_args()
        config = process_config(args.config)
    except ValueError:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.log_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create an instance of the model you want
    model = S2L1CToL2AModel(config)
    # load model if exists
    model.load(sess)
    # create workflow to execute in online data generator
    s2_l1c_l2a_workflow = S2L1CToL2AWorkflow(config)
    # create your data generator
    data = OnlineEODataGenerator(config, s2_l1c_l2a_workflow)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = FCNTrainer(sess, model, data, config, logger)
    # here you train your model
    trainer.train()
    # freeze weights
    model.freeze_graph()
    with tf.Session() as sess:
        test_logger = Logger(sess, config)
        # predictor class
        predictor = S2L1CToL2APredict(config, data, test_logger)
        predictor.predict()


if __name__ == '__main__':
    main()
