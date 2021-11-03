from .tflite import (
    CPUBackend,
    GPUBackend
)
from .openvino import (
    VPUBackend
)
from .interface import get_backend