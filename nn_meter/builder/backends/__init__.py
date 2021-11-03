# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from .tflite import (
    CPUBackend,
    GPUBackend
)
from .openvino import (
    VPUBackend
)
from .interface import get_backend