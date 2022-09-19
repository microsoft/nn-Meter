from nn_meter.builder.backends import connect_backend
from nn_meter.builder import builder_config, convert_models

workspace = "/data1/jiahang/working/pixel4_int8_workspace"
builder_config.init(workspace)
backend = connect_backend(backend_name='tflite_cpu_int8')

# kernels_info = builder_config.get("KERNELS", "predbuild")

kernels_info = ["concat_block"]
for kernel_type in kernels_info:
    # init_sample_num = kernels_info[kernel_type]["INIT_SAMPLE_NUM"]
    # models = generate_config_sample(kernel_type, init_sample_num, mark='prior', sampling_mode='prior')
    models = convert_models(backend, f"{workspace}/predictor_build/results/{kernel_type}_prior.json", broken_point_mode=True)
