# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import shutil
import logging
from ..interface import BaseBackend
from nn_meter.utils.path import get_filename_without_ext
logging = logging.getLogger("nn-Meter")


class TFLiteBackend(BaseBackend):
    parser_class = None
    profiler_class = None

    def update_configs(self):
        """update the config parameters for TFLite platform
        """
        super().update_configs()
        self.profiler_kwargs.update({
            'dst_graph_path': self.configs['REMOTE_MODEL_DIR'],
            'benchmark_model_path': self.configs['BENCHMARK_MODEL_PATH'],
            'serial': self.configs['DEVICE_SERIAL'],
            'dst_kernel_path': self.configs['KERNEL_PATH']
        })

    def convert_model(self, model_path, save_path, input_shape=None):
        """convert the Keras model instance to ``.tflite`` and return model path
        """
        import tensorflow as tf
        model_name = get_filename_without_ext(model_path)
        model = tf.keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        converted_model = os.path.join(save_path, model_name + '.tflite')
        open(converted_model, 'wb').write(tflite_model)
        shutil.rmtree(model_path)
        return converted_model

    def test_connection(self):
        """check the status of backend interface connection, ideally including open/close/check_healthy...
        """
        from ppadb.client import Client as AdbClient
        client = AdbClient(host="127.0.0.1", port=5037)
        if self.configs['DEVICE_SERIAL']:
            device = client.device(self.configs['DEVICE_SERIAL'])
        else:
            device = client.devices()[0]
        logging.keyinfo(device.shell("echo hello backend !"))
