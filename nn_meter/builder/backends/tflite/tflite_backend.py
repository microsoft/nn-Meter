# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import tensorflow as tf

from ..interface import BaseBackend
from nn_meter.builder.utils import get_tensor_by_shapes


class TFLiteBackend(BaseBackend):
    parser_class = None
    runner_class = None

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
        # build input shapes
        model(get_tensor_by_shapes(shapes))
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        graph_path = os.path.join(self.model_dir, model_name + '.tflite')
        open(graph_path, 'wb').write(tflite_model)
        self.runner.load_graph(graph_path, os.path.join(self.remote_model_dir, model_name + '.tflite'))
        return self.parser.parse(self.runner.run()).latency
