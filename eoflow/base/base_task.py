from eoflow.base import Configurable

class BaseTask(Configurable):
    def __init__(self, model, config_specs):
        super().__init__(config_specs)

        self.model = model

    def run(self):
        raise NotImplementedError
