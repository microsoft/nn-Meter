# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys
import yaml
import importlib


__BUILTIN_BACKENDS__ = {
    "tflite_cpu": {
        "class_module": "nn_meter.builder.backends.tflite",
        "class_name": "TFLiteCPUBackend"
    },
    "tflite_gpu": {
        "class_module": "nn_meter.builder.backends.tflite",
        "class_name": "TFLiteGPUBackend"
    },
    "openvino_vpu": {
        "class_module": "nn_meter.builder.backends.openvino",
        "class_name": "OpenVINOVPUBackend"
    },
    "debug_backend": {
        "class_module": "nn_meter.builder.backends.interface",
        "class_name": "DebugBackend"
    }
}

__user_config_folder__ = os.path.expanduser('~/.nn_meter/config')
__registry_cfg_filename__ = 'registry.yaml'
__REG_BACKENDS__ = {}
if os.path.isfile(os.path.join(__user_config_folder__, __registry_cfg_filename__)):
    with open(os.path.join(__user_config_folder__, __registry_cfg_filename__), 'r') as fp:
        registry_modules = yaml.load(fp, yaml.FullLoader)
    if "backends" in registry_modules:
        __REG_BACKENDS__ = registry_modules["backends"]


class BaseBackend:
    """
    the base backend class to instantiate a backend instance. If users want to implement their own backend,
    the customized Backend should inherit this class.

    @params:

    profiler_class: a subclass inherit form `nn_meter.builder.backend.BaseProfiler` to specify the running command of
        the backend. A profiler contains commands to push the model to mobile device, run the model on the mobile device,
        get stdout from the mobile device, and related operations. In the implementation of a profiler, an interface of
        ``Profiler.profile()`` is required.
    
    parser_class: a subclass inherit form `nn_meter.builder.backend.BaseParser` to parse the profiled results.
        A parser parses the stdout from devices profiler and get required metrics. In the implementation of a parser, interface
        of `Parser.parse()` and property of `Parser.results()` are required.
    """
    profiler_class = None
    parser_class = None

    def __init__(self, configs):
        """ class initialization with required configs
        """
        self.configs = configs
        self.update_configs()
        if self.parser_class:
            self.parser = self.parser_class(**self.parser_kwargs)
        if self.profiler_class:
            self.profiler = self.profiler_class(**self.profiler_kwargs)

    def update_configs(self):
        """ update the config parameters for the backend
        """
        self.parser_kwargs = {}
        self.profiler_kwargs = {}
    
    def convert_model(self, model_path, save_path, input_shape=None):
        """ convert the Keras model instance to the type required by the backend inference.

        @params:
        
        model_path: the path of model waiting to profile
        
        save_path: folder to save the converted model
        
        input_shape: the shape of input tensor for inference, a random tensor according to the shape will be 
            generated and used
        """
        # convert model and save the converted model to path `converted_model`
        converted_model = model_path
        return converted_model

    def profile(self, converted_model, metrics = ['latency'], input_shape = None, **kwargs):
        """
        run the model on the backend, return required metrics of the running results. nn-Meter only support latency
        for metric by now. Users may provide other metrics in their customized backend.

        @params:

        converted_model: the model path in type of backend required
        
        metrics: a list of required metrics name. Defaults to ['latency']
        
        """
        return self.parser.parse(self.profiler.profile(converted_model, **kwargs)).results.get(metrics)

    def profile_model_file(self, model_path, save_path, input_shape = None, metrics = ['latency'], **kwargs):
        """ load model by model file path, convert model file, and run ``self.profile()``
        @params:

        model_path: the path of model waiting to profile
        
        save_path: folder to save the converted model
        
        input_shape: the shape of input tensor for inference, a random tensor according to the shape will be 
            generated and used
        """
        converted_model = self.convert_model(model_path, save_path, input_shape)
        res = self.profile(converted_model, metrics, input_shape, **kwargs)
        return res

    def test_connection(self):
        """ check the status of backend interface connection.
        """
        pass


class BaseProfiler:
    """
    Specify the profiling command of the backend. A profiler contains commands to push the model to mobile device, run the model 
    on the mobile device, get stdout from the mobile device, and related operations. 
    """
    def profile(self):
        """ Main steps of ``Profiler.profile()`` includes 1) push the model file to edge devices, 2) run models in required times
        and get back running results. Return the running results on edge device.
        """
        output = ''
        return output


class BaseParser:
    """
    Parse the profiled results. A parser parses the stdout from devices runner and get required metrics.
    """
    def parse(self, content):
        """ A string parser to parse profiled results value from the standard output of devices runner. This method should return the instance
        class itself.

        @params
        
        content: the standard output from device       
        """
        return self

    @property
    def results(self):
        """ warp the parsed results by ``ProfiledResults`` class from ``nn_meter.builder.backend_meta.utils`` and return the parsed results value.
        """
        pass

class DebugBackend(BaseBackend):
    """ For debug use when there is no backend available. All latency value are randomly generated.
    """
    
    def profile(self, converted_model, metrics = ['latency'], input_shape = None, **kwargs):
        import random
        from nn_meter.builder.backend_meta.utils import Latency, ProfiledResults
        latency = Latency(random.randrange(0, 10000) / 100, random.randrange(0, 1000) / 1000) 
        return ProfiledResults({'latency': latency}).get(metrics)

    def test_connection(self):
        """ check the status of backend interface connection.
        """
        import logging
        logging.info("hello backend !")


def connect_backend(backend_name):
    """ 
    Return the required backend class, and feed params to the backend. Supporting backend: tflite_cpu, tflite_gpu, openvino_vpu.
    
    Available backend and corresponding configs: 
    - For backend based on TFLite platform: {
        'REMOTE_MODEL_DIR': path to the folder (on mobile device) where temporary models will be copied to.
        'BENCHMARK_MODEL_PATH': path (on android device) where the binary file `benchmark_model` is deployed.
        'DEVICE_SERIAL': if there are multiple adb devices connected to your host, you need to provide the \\
                         corresponding serial id. Set to '' if there is only one device connected to your host.
        'KERNEL_PATH': path (on mobile device) where the kernel implementations will be dumped.
    }
    - For backend based on OpenVINO platform: {
        'OPENVINO_ENV': path to openvino virtualenv (./docs/requirements/openvino_requirements.txt is provided)
        'OPTIMIZER_PATH': path to openvino optimizer
        'OPENVINO_RUNTIME_DIR': directory to openvino runtime
        'DEVICE_SERIAL': serial id of the device
        'DATA_TYPE': data type of the model (e.g., fp16, fp32)
    }
    
    The config can be declared and modified after create a workspace. Users could follow guidance from ./docs/builder/backend.md
    
    @params:
    backend_name: name of backend (subclass instance of `BaseBackend`). 
    """
    if backend_name in __REG_BACKENDS__:
        backend_info = __REG_BACKENDS__[backend_name]
        sys.path.append(backend_info["package_location"])
    elif backend_name in __BUILTIN_BACKENDS__:
        backend_info = __BUILTIN_BACKENDS__[backend_name]
    else:
        raise ValueError(f"Unsupported backend name: {backend_name}. Please register the backend first.")

    module = backend_info["class_module"]
    name = backend_info["class_name"]
    backend_module = importlib.import_module(module)   
    backend_cls = getattr(backend_module, name)

    # load configs from workspace
    from nn_meter.builder import builder_config
    configs = builder_config.get_module('backend')
    return backend_cls(configs)


def list_backends():
    """ list all backends supported by nn-Meter, including builtin backends and registered backends
    """
    return list(__BUILTIN_BACKENDS__.keys()) + ["* " + item for item in list(__REG_BACKENDS__.keys())]
