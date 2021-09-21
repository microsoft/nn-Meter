import importlib


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
    'tflite_gpu': {
        'MODEL_DIR': '',
        'REMOTE_MODEL_DIR': '',
        'KERNEL_PATH': '',
        'BENCHMARK_MODEL_PATH': '',
        'DEVICE_SERIAL': '',
    },
    'tflite_cpu': {
        'MODEL_DIR': '',
        'REMOTE_MODEL_DIR': '',
        'KERNEL_PATH': '',
        'BENCHMARK_MODEL_PATH': '',
        'DEVICE_SERIAL': '',
    },
    'vpu': {
        'MOVIDIUS_ENV': '',
        'OPTIMIZER_PATH': '',
        'TMP_DIR': '',
        'OPENVINO_RUNTIME_DIR': '',
        'DEVICE_SERIAL': '',
        'DATA_TYPE': '',
    },
    ```
    """
    if isinstance(backend, str):
        backend_module = importlib.import_module(BACKENDS[backend] + '.base')
        backend_cls = getattr(backend_module, 'Backend')(params)
    else:
        backend_cls = backend
    return backend_cls(params)


def register_backend(name, engine):
    BACKENDS[name] = engine
