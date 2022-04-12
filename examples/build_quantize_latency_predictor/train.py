from nn_meter.builder.backends import connect_backend
from nn_meter.builder import builder_config, convert_models, profile_models
from nn_meter.builder.utils import merge_info
from nn_meter.builder.kernel_predictor_builder import generate_config_sample
from nn_meter.builder.kernel_predictor_builder import build_predictor_by_data
    

workspace = "/data1/jiahang/working/pixel4_int8_workspace"
builder_config.init(workspace)
backend = connect_backend(backend_name='tflite_cpu_int8')

kernels_info = builder_config.get("KERNELS", "predbuild")
kernels = [
    "add", "addrelu", "avgpool", "bn", "bnrelu", "channelshuffle", "concat",
    "fc", "global-avgpool", "hswish", "maxpool", "relu", "se", "split"
]

for kernel_type in kernels:
    # prior prediction
    kernel_data = (f'{workspace}/predictor_build/results/{kernel_type}_prior.json',
                   f'{workspace}/predictor_build/results/profiled_{kernel_type}.json')
    # use current sampled data to build regression model, and locate data with large errors in testset
    predictor, acc10, error_configs = build_predictor_by_data(kernel_type, kernel_data, "tflite_cpu", error_threshold=0.1, mark='prior',
                                                              save_path=f"{workspace}/predictor_build/collection/")
    print(f'Iteration 0: acc10 {acc10}, error_configs number: {len(error_configs)}')
    print("\n")

    # # ####---------- finegrained1 ----------####
    # # finegrained sampling and profiling for large error data
    # finegrained_sample_num = kernels_info[kernel_type]["FINEGRAINED_SAMPLE_NUM"]
    # models = generate_config_sample(kernel_type, 2, mark='finegrained1', sampling_mode='finegrained', configs=error_configs)
    # models = convert_models(backend, f"{workspace}/predictor_build/results/{kernel_type}_finegrained1.json", broken_point_mode=True)


# kernel_type = "conv-bn-relu"
# config = merge_info(new_info=f'{workspace}/predictor_build/results/{kernel_type}_prior1.json',
#                     prev_info=f'{workspace}/predictor_build/results/{kernel_type}_prior2.json')
# config = merge_info(new_info=f'{workspace}/predictor_build/results/{kernel_type}_finegrained1.json',
#                     prev_info=config)
# config = merge_info(new_info=f'{workspace}/predictor_build/results/{kernel_type}_finegrained1-1.json',
#                     prev_info=config)
# config = merge_info(new_info=f'{workspace}/predictor_build/results/{kernel_type}_newsampling.json',
#                     prev_info=config)

# latency = merge_info(new_info=f'{workspace}/predictor_build/results/profiled_{kernel_type}.json',
#                      prev_info=f'{workspace}/predictor_build/results/profiled_{kernel_type}_newsampling.json')
# from nn_meter.builder.backend_meta.utils import read_profiled_results
# kernel_data = (config, read_profiled_results(latency))

# # use current sampled data to build regression model, and locate data with large errors in testset
# from nn_meter.builder.kernel_predictor_builder import build_predictor_by_data
# predictor, acc10, error_configs = build_predictor_by_data(kernel_type, kernel_data, "tflite_cpu", error_threshold=0.1, mark='newsampling',
#                                                           save_path=f"{workspace}/predictor_build/results/")
# print(f'Iteration 1: acc10 {acc10}, error_configs number: {len(error_configs)}')
# print("\n")
