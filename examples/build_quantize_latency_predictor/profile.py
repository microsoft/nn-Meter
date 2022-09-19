import logging
import os
from nn_meter.builder.backends import connect_backend
from nn_meter.builder import builder_config, convert_models, profile_models
from nn_meter.builder.kernel_predictor_builder import generate_config_sample


workspace = "/data1/jiahang/working/pixel4_int8_workspace"
builder_config.init(workspace)
backend = connect_backend(backend_name='tflite_cpu_int8')

kernels = ["maxpool_block", "avgpool_block"]

for kernel_type in kernels:
    profile_models(
        backend,
        models=f"{workspace}/predictor_build/results/{kernel_type}_prior.json",
        mode='predbuild', 
        have_converted=True,
        save_name=f"profiled_{kernel_type}.json"
    )
