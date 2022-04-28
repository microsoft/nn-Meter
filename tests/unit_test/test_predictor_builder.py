# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Note: this script could only run after setting nn-Meter builder up.

# initialize builder config with workspace
from nn_meter.builder import builder_config
builder_config.init("/path/to/workspace/") # initialize builder config with workspace

# build latency predictor for kernel
from nn_meter.builder import build_predictor_for_kernel
kernel_type = "conv-bn-relu"
backend = "tflite_cpu"

predictor, data = build_predictor_for_kernel(
    kernel_type, backend, init_sample_num = 5, finegrained_sample_num = 10, iteration = 5, error_threshold = 0.1
)