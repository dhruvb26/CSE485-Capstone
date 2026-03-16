"""Base class for all dataset handlers."""


class BaseDatasetHandler:

    def __init__(self, name, args):
        self.name = name
        self.args = args
        self.setup_dataset()

    def setup_dataset(self):
        raise NotImplementedError

    def get_instances(self):
        raise NotImplementedError
