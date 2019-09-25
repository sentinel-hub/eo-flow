from eoflow.base import Configurable

class BaseInput(Configurable):
    def get_dataset(self):
        raise NotImplementedError
