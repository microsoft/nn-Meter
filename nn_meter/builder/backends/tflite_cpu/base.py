from nn_meter.builder.backends.tflite_gpu.base import Backend as GPUBackend
from .parser import TFLiteCPUParser
from .runner import TFLiteCPURunner


class Backend(GPUBackend):
    parser_class = TFLiteCPUParser
    runner_class = TFLiteCPURunner
