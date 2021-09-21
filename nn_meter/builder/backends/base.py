from nn_meter.builder.utils.latency import Latency
from utils.path import get_filename_without_ext
import tensorflow as tf


class BaseBackend:
    parser_class = None
    runner_class = None

    def __init__(self, params):
        self.params = params
        self.get_params()
        self.parser = self.parser_class(**self.parser_kwargs)
        self.runner = self.runner_class(**self.runner_kwargs)

    def get_params(self):
        self.parser_kwargs = {}
        self.runner_kwargs = {}

    def profile(self, model, model_name, input_shape=None):
        return Latency()

    def profile_model_file(self, model_path, shapes=None):
        model_name = get_filename_without_ext(model_path)
        model = tf.keras.models.load_model(model_path)
        return self.profile(model, model_name, shapes)
