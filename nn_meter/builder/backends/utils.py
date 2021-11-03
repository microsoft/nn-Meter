import importlib

# TODO: test the connection
# command: nn-meter device/backend test
BACKENDS = {
    'tflite_gpu': 'nn_meter.builder.backends.tflite_gpu',
    'tflite_cpu': 'nn_meter.builder.backends.tflite_cpu',
    'vpu': 'nn_meter.builder.backends.vpu',
}


def get_backend(backend, params):
    """
    @params:

    backend: str of path to module or subclass of `BaseBackend`

    Available backend and required params:

    ```
    tflite_gpu: {
        'MODEL_DIR': '',  # directory on host to save temporary tflite models
        'REMOTE_MODEL_DIR': '',  # directory on mobile phone to place models
        'KERNEL_PATH': '',  # directory on mobile phone where kernel code files will be generated
        'BENCHMARK_MODEL_PATH': '',  # path to bin of `benchmark_model`
        'DEVICE_SERIAL': '',  # serial id of the device. set to '' if there is only one device connected to your host.
    },
    tflite_cpu: {
        'MODEL_DIR': '',
        'REMOTE_MODEL_DIR': '',
        'KERNEL_PATH': '',
        'BENCHMARK_MODEL_PATH': '',
        'DEVICE_SERIAL': '',
    },
    vpu: {
        'MOVIDIUS_ENV': '',  # path to movidius virtualenv (movidius_requirements.txt is provided)
        'OPTIMIZER_PATH': '',  # path to openvino optimizer
        'TMP_DIR': '',  # tmp directory where temp model and profiling results will be generated
        'OPENVINO_RUNTIME_DIR': '',  # directory to openvino runtime
        'DEVICE_SERIAL': '',  # serial id of the device
        'DATA_TYPE': '',  # data type of the model (e.g., fp16, fp32)
    },
    ```
    """
    if isinstance(backend, str):
        backend_module = importlib.import_module(BACKENDS[backend] + '.base')
        backend_cls = getattr(backend_module, 'Backend')
    else:
        backend_cls = backend
    return backend_cls(params)


def register_backend(name, engine):
    BACKENDS[name] = engine
