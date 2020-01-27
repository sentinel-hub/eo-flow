from . import Configurable, BaseInput
from ..utils import parse_classname


class BaseTask(Configurable):
    def __init__(self, model, config_specs):
        super().__init__(config_specs)

        self.model = model

    @staticmethod
    def parse_input(input_config):
        """ Builds the input dataset using the provided configuration. """

        classname, config = input_config.classname, input_config.config

        cls = parse_classname(classname)
        if not issubclass(cls, BaseInput):
            raise ValueError("Data input class does not inherit from BaseInput.")

        model_input = cls(config)

        return model_input.get_dataset()

    def run(self):
        """Executes the task."""

        raise NotImplementedError
