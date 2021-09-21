from backends.base import BaseBackend
from .parser import TFLiteGPUParser
from .runner import TFLiteGPURunner
import tensorflow as tf
import os
from nn_meter.builder.utils.path import get_filename


class Backend(BaseBackend):
    parser_class = TFLiteGPUParser
    runner_class = TFLiteGPURunner

    def get_params(self):
        super().get_params()
        self.runner_kwargs.update({
            'dst_kernel_path': self.params['KERNEL_PATH'],
            'serial': self.params['DEVICE_SERIAL'],
            'benchmark_model_path': self.params['BENCHMARK_MODEL_PATH'],
        })
        self.model_dir = self.params['MODEL_DIR']
        self.remote_model_dir = self.params['REMOTE_MODEL_DIR']

    def profile(self, model, model_name, shapes=None):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        graph_path = os.path.join(self.model_dir, model_name + '.tflite')
        open(graph_path, 'wb').write(tflite_model)
        self.runner.load_graph(graph_path, os.path.join(self.remote_model_dir, model_name + '.tflite'))
        return self.parser.parse(self.runner.run()).latency
