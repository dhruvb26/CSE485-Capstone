"""Base class for all model handlers."""


class BaseModelHandler:

    def __init__(self, name, args):
        self.name = name
        self.args = args
        self.setup_model()

    def setup_model(self):
        raise NotImplementedError

    def get_model_outputs(self, inputs, ground_truth):
        raise NotImplementedError
