from . import Configurable


class BaseInput(Configurable):
    def get_dataset(self):
        """Builds and returns a tensorflow Dataset object for reading the data."""

        raise NotImplementedError
