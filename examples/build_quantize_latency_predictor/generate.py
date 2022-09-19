from nn_meter.builder.backends import connect_backend
from nn_meter.builder import builder_config, convert_models, profile_models
from nn_meter.builder.kernel_predictor_builder import generate_config_sample


workspace = "/data1/jiahang/working/pixel4_int8_workspace"
builder_config.init(workspace)
backend = connect_backend(backend_name='tflite_cpu_int8')


kernels_info = builder_config.get("KERNELS", "predbuild")

kernels = [
    "conv-bn-relu"
    ]
for kernel_type in kernels:
    init_sample_num = kernels_info[kernel_type]["INIT_SAMPLE_NUM"]
    models = generate_config_sample(kernel_type, init_sample_num, mark='test', sampling_mode='prior')
    profile_models(
        backend,
        models=f"{workspace}/predictor_build/results/{kernel_type}_test.json",
        mode='predbuild', 
        have_converted=False,
        save_name=f"profiled_{kernel_type}_test.json"
    )
