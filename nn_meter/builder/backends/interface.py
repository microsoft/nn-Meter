# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import importlib
from typing import List
import tensorflow as tf
from nn_meter.builder.utils import builder_config as config
from nn_meter.utils.path import get_filename_without_ext


BACKENDS_PATH = {
    'tflite_gpu': ['nn_meter.builder.backends', 'TFLiteGPUBackend'],
    'tflite_cpu': ['nn_meter.builder.backends', 'TFLiteCPUBackend'],
    'openvino_vpu': ['nn_meter.builder.backends', 'OpenVINOVPUBackend'],
}


class BaseBackend:
    """
    the base backend class to instantiate a backend instance. If users want to implement their own backend,
    the customized Backend should inherit this class.

    @params:

    runner_class: the class to specify the running command of the backend. A runner contains commands to push
        the model of testcases to mobile device, run the model on the mobile device, get stdout from the mobile
        device, and related operations. In the implementation of a runner, an interface of ``Runner.run()`` is
        required.
    
    parser_class: the class of the profiled results parser. A parser parses the stdout from runner and get 
        required metrics. In the implementation of a parser, an interface of ``Parser.parse()`` is required.
    """
    runner_class = None
    parser_class = None

    def __init__(self, name, params):
        """class initialization with required params
        """
        self.name = name
        self.params = params
        self.update_params()
        self.parser = self.parser_class(**self.parser_kwargs)
        self.runner = self.runner_class(**self.runner_kwargs)

    def update_params(self):
        """update the config parameters for the backend
        """
        self.parser_kwargs = {}
        self.runner_kwargs = {}
    
    def convert_model(self, model, model_name, input_shape=None):
        """convert the Keras model instance to the type required by the backend inference
        """
        return model

    def profile(self, model, model_name, input_shape=None, metrics=['latency']):
        """
        convert the model to the backend platform and run the model on the backend, return required metrics 
        of the running results. We only support latency for metric by now.

        @params:
        
        model: the Keras model waiting to profile
        
        model_name: the name of the model
        
        input_shape: the shape of input tensor for inference, a random tensor according to the shape will be 
            generated and used
        
        metrics: a list of required metrics name. Defaults to ['latency']
        
        """
        converted_model = self.convert_model(model, model_name, input_shape)
        return self.parser.parse(self.runner.run(converted_model)).results.get(metrics)

    def profile_model_file(self, model_path, input_shape=None, metrics=['latency']):
        """load model by model file path and run ``self.profile()``
        """
        model_name = get_filename_without_ext(model_path)
        model = tf.keras.models.load_model(model_path)
        return self.profile(model, model_name, input_shape, metrics)

    def test_connection(self):
        """check the status of backend interface connection, ideally including open/close/check_healthy...
        TODO: add command line interface: nn-meter device/backend connect
        """
        pass


def connect_backend(backend):
    """ 
    Return the required backend class, and feed params to the backend
    
    @params:
    backend: str of path to module or subclass of `BaseBackend`

    workspace_path: path to the workspace. Users could refer to docs/builder/backend.md for further information. 
        Available backend and required params: 
        TODO: refine here, remove ws path and load it in config

        - For backend based on TFLite platform: {
            'MODEL_DIR': '',  # path to the folder (on host device) where temporary models will be generated.
            'REMOTE_MODEL_DIR': '',  # path to the folder (on mobile device) where temporary models will be copied to.
            'KERNEL_PATH': '',  # path (on mobile device) where the kernel implementations will be dumped.
            'BENCHMARK_MODEL_PATH': '',  # path (on android device) where the binary file `benchmark_model` is deployed.
            'DEVICE_SERIAL': '',  # if there are multiple adb devices connected to your host, you need to provide the \\
                                  # corresponding serial id. Set to '' if there is only one device connected to your host.
        }
        - For backend based on OpenVINO platform: {
            'OPENVINO_ENV': '',  # path to openvino virtualenv (./docs/requirements/openvino_requirements.txt is provided)
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
        backend_name = backend
    else:
        backend_cls = backend
        backend_name = backend.name
    params = config.get_module('backend')
    return backend_cls(backend_name, params)


def register_backend(backend_name, engine_path, cls_name):
    """
    Register a customized backend. The engine_path, cls_name will be used as:
        'from <engine_path> import <cls_name>'

    @params:

    backend_name: alias name of the new backend

    engine_path: the path to define the customized backend

    cls_name: class name of the customized backend class, should be a subclass of BaseBackend
    """
    BACKENDS_PATH[backend_name] = [engine_path, cls_name]
