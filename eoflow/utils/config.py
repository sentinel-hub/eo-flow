from bunch import Bunch
import logging
import json
import os

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def process_config(json_file):
    logging.debug("Reading config from .json file")
    config, _ = get_config_from_json(json_file)
    config.log_dir = os.path.join(config.exp_dir, config.exp_name, "logs/")
    config.checkpoint_dir = os.path.join(config.exp_dir, config.exp_name, "checkpoints/")
    return config
