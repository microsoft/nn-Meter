from .runner import VPURunner
from .parser import VPUParser
from backends.base import BaseBackend
from utils.converters import keras_model_to_frozen_pb
from utils.patcher_wrapper import patch_frozen_pb
from utils.path import get_filename_without_ext
import os
import tensorflow as tf


class Backend(BaseBackend):
    parser_class = VPUParser
    runner_class = VPURunner

    def get_params(self):
        super().get_params()
        self.runner_kwargs.update({
            'venv': self.params['MOVIDIUS_ENV'],
            'optimizer': self.params['OPTIMIZER_PATH'],
            'runtime_dir': self.params['OPENVINO_RUNTIME_DIR'],
            'serial': self.params['DEVICE_SERIAL'],
            'data_type': self.params['DATA_TYPE'],
        })
        self.tmp_dir = self.params['TMP_DIR']
        self.venv = self.params['MOVIDIUS_ENV']

    def profile(self, model, model_name, shapes):
        model_tmp_dir = os.path.join(self.tmp_dir, model_name)
        pb_path, _ = keras_model_to_frozen_pb(model, model_tmp_dir, model_name, shapes)
        patched_pb_path = patch_frozen_pb(pb_path, os.path.join(self.venv, 'bin/python'))
        self.runner.load_graph(patched_pb_path, model_tmp_dir)
        return self.parser.parse(self.runner.run(shapes)).latency

    def profile_model_file(self, model_path, shapes):
        model_name = get_filename_without_ext(model_path)
        model = tf.keras.models.load_model(model_path)
        return self.profile(model, model_name, shapes)
