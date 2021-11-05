# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import importlib
import tensorflow as tf

from nn_meter.builder.utils.latency import Latency
from nn_meter.utils.path import get_filename_without_ext


BACKENDS_PATH = {
    'tflite_gpu': ['nn_meter.builder.backends', 'GPUBackend'],
    'tflite_cpu': ['nn_meter.builder.backends', 'CPUBackend'],
    'openvino_vpu': ['nn_meter.builder.backends', 'VPUBackend'],
}


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

    def test_connection(self):
        # TODO: test the connection (open/close/check_healthy)
        # add command line interface: nn-meter device/backend test
        pass


def get_backend(backend, params):
    """
    @params:

    - backend: str of path to module or subclass of `BaseBackend`

    - params: Available backend and required params:

        - tflite: {
            'MODEL_DIR': '',  # directory on host to save the generated tflite models
            'REMOTE_MODEL_DIR': '',  # directory on mobile phone to place models
            'KERNEL_PATH': '',  # directory on mobile phone where kernel code files will be generated
            'BENCHMARK_MODEL_PATH': '',  # path to bin of `benchmark_model`
            'DEVICE_SERIAL': '',  # serial id of the device. set to '' if there is only one device connected to your host.
        }
        - openvino: {
            'OPENVINO_ENV': '',  # path to openvino virtualenv (openvino_requirements.txt is provided)
            'OPTIMIZER_PATH': '',  # path to openvino optimizer
            'TMP_DIR': '',  # tmp directory where temp model and profiling results will be generated
            'OPENVINO_RUNTIME_DIR': '',  # directory to openvino runtime
            'DEVICE_SERIAL': '',  # serial id of the device
            'DATA_TYPE': '',  # data type of the model (e.g., fp16, fp32)
        }
    """
    if isinstance(backend, str):
        module, attr = BACKENDS_PATH[backend]
        backend_module = importlib.import_module(module)
        backend_cls = getattr(backend_module, attr)
    else:
        backend_cls = backend
    return backend_cls(params)


def register_backend(backend_name, engine_path, cls_name):
    """
    Register a customized backend. The engine_path, cls_name will be used as:
        'from <engine_path> import <cls_name>'

    @params:

    - backend_name: alias name of the new backend

    - engine_path: the path to define the customized backend

    - cls_name: class name of the customized backend class, should be a subclass of BaseBackend
    """
    BACKENDS_PATH[backend_name] = [engine_path, cls_name]
