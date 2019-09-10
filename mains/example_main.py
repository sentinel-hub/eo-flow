import tensorflow as tf

from eoflow.data_loader import ExampleDataGenerator
from eoflow.models import ExampleModel
from eoflow.trainers import ExampleTrainer
from eoflow.utils import process_config
from eoflow.utils import create_dirs
from eoflow.utils import Logger
from eoflow.utils import get_args


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
    model = ExampleModel(config)
    # load model if exists
    model.load(sess)
    # create your data generator
    data = ExampleDataGenerator(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = ExampleTrainer(sess, model, data, config, logger)
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
