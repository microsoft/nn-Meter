# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import tensorflow as tf

from ..interface import BaseBackend
from .utils import keras_model_to_frozenpb
from .frozenpb_patcher import patch_frozenpb
from nn_meter.utils.path import get_filename_without_ext


class OpenVINOBackend(BaseBackend):
    parser_class = None
    runner_class = None

    def get_params(self):
        super().get_params()
        self.runner_kwargs.update({
            'venv': self.params['OPENVINO_ENV'],
            'optimizer': self.params['OPTIMIZER_PATH'],
            'runtime_dir': self.params['OPENVINO_RUNTIME_DIR'],
            'serial': self.params['DEVICE_SERIAL'],
            'data_type': self.params['DATA_TYPE'],
        })
        self.tmp_dir = self.params['TMP_DIR']
        self.venv = self.params['OPENVINO_ENV']

    def profile(self, model, model_name, shapes):
        model_tmp_dir = os.path.join(self.tmp_dir, model_name)
        pb_path, _ = keras_model_to_frozenpb(model, model_tmp_dir, model_name, shapes)
        patched_pb_path = patch_frozenpb(pb_path, os.path.join(self.venv, 'bin/python'))
        self.runner.load_graph(patched_pb_path, model_tmp_dir)
        return self.parser.parse(self.runner.run(shapes)).latency

    def profile_model_file(self, model_path, shapes):
        model_name = get_filename_without_ext(model_path)
        model = tf.keras.models.load_model(model_path)
        return self.profile(model, model_name, shapes)
