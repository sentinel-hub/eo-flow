from abc import ABC
import inspect
from marshmallow import Schema
from munch import Munch
import json

def dict_to_munch(obj):
    """ Recursively convert a dict to Munch. (there is a Munch.from_dict method, but it's not python3 compatible)
    """
    if isinstance(obj, list):
        return [dict_to_munch(element) for element in obj]
    if isinstance(obj, dict):
        return Munch({k: dict_to_munch(v) for k, v in obj.items()})
    return obj


class Configurable(ABC):
    """Base class for all configurable objects."""

    def __init__(self, config_specs):
        self.schema = self.initialize_schema()
        self.config = self._prepare_config(config_specs)

    @classmethod
    def initialize_schema(cls):
        """ PipelineSchema's child class should be an internal class of any Pipeline's child class. This method collects
        this class if it is specified. Otherwise it takes default PipelineSchema class
        """
        for item in vars(cls).values():
            if inspect.isclass(item) and issubclass(item, Schema):
                return item()

        if len(cls.__bases__) > 1:
            raise RuntimeError('Class does not have a defined schema however it inherits from multiple '
                               'classes. Which one should schema be inherited from?')

        parent_class = cls.__bases__[0]

        if parent_class is Configurable:
            raise NotImplementedError('Configuration schema not provided.')

        return parent_class.initialize_schema()

    def _prepare_config(self, config_specs):
        """ Collects and validates configuration dictionary
        """

        # if config_specs is a path
        if isinstance(config_specs, str):
            with open(config_specs, 'r') as config_file:
                config_specs = json.load(config_file)

        return Config(self.schema.load(config_specs))


class Config(Munch):
    """ Class for handling configuration files
    """
    def __init__(self, config):
        config = dict_to_munch(config)

        super().__init__(config)
