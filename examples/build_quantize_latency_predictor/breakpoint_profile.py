import os
import json
import logging
import time
from nn_meter.builder import builder_config
from nn_meter.builder.kernel_predictor_builder.data_sampler.utils import generate_model_for_kernel
from nn_meter.builder.utils import save_profiled_results
from nn_meter.builder.backends import connect_backend

workspace = "/data1/jiahang/working/pixel4_int8_workspace"
builder_config.init(workspace)
backend = connect_backend(backend_name='tflite_cpu_int8')

def profile_models(backend, models, kernel_type, mode = 'ruletest', metrics = ["latency"], save_name = None,
                   **kwargs):
    """ run models with given backend and return latency of testcase models

    @params:

    backend (subclass instance of BaseBackend): applied backend instance

    models (str or dict): the Dict of models or the path of the json file about models information 

    mode (str): the mode for running models, including ['ruletest', 'predbuild']

    metrics (list): required metrics to report. We only support latency for metric by now.

    save_name (str): the save name to store profiled results. The whole path should be "<workspace>/<mode-folder>/results/<save-name>"

    have_converted (boolean): if the model have been converted to the needed format by backend, the model will not be converted
        before profiling. The model path of `model['converted_model']` will be profiled on device directly. The conversion of
        model could be done by appling `nn_meter.builder.convert_models`

    **kwargs: arguments for profiler
    """
    if isinstance(models, str):
        with open(models, 'r') as fp:
            models = json.load(fp)

    ws_mode_path = builder_config.get('MODEL_DIR', mode)
    model_save_path = os.path.join(ws_mode_path, 'models')
    os.makedirs(model_save_path, exist_ok=True)
    info_save_path = os.path.join(ws_mode_path, "results")
    os.makedirs(info_save_path, exist_ok=True)

    # profile models and get metric results
    count = 0
    detail = builder_config.get('DETAIL', mode)
    save_name = save_name or "profiled_results.json"
    logging.info("Profiling ...")
    for _, modules in models.items():
        for id, model in modules.items():
            model_path = model['model']
            print(model_path)
            if os.path.isfile(model_path + ".tflite"):
                try:
                    profiled_res = backend.profile(model_path + ".tflite", metrics, model['shapes'], **kwargs)
                    for metric in metrics:
                        model[metric] = profiled_res[metric]
                    time.sleep(0.2)
                    count += 1
                except Exception as e:
                    open(os.path.join(info_save_path, "profile_error.log"), 'a').write(f"{id}: {e}\n")
            elif os.path.exists(model_path):
                try:
                    profiled_res = backend.profile_model_file(model_path, model_save_path, model['shapes'], metrics, **kwargs)
                    for metric in metrics:
                        model[metric] = profiled_res[metric]
                    time.sleep(0.2)
                    count += 1
                except Exception as e:
                    open(os.path.join(info_save_path, "profile_error.log"), 'a').write(f"{id}: {e}\n")
            else:
                try:
                    model_config = model['config']
                    generate_model_for_kernel(
                        kernel_type, model_config, save_path=model_path,
                        implement='tensorflow', batch_size=1
                    )
                    profiled_res = backend.profile_model_file(model_path, model_save_path, model['shapes'], metrics, **kwargs)
                    for metric in metrics:
                        model[metric] = profiled_res[metric]
                    time.sleep(0.2)
                    count += 1
                except Exception as e:
                    open(os.path.join(info_save_path, "generate_error.log"), 'a').write(f"{id}: {e}\n")

            # save information to json file for per 50 models
            if count > 0 and count % 50 == 0:
                save_profiled_results(models, os.path.join(info_save_path, save_name), detail, metrics)
                logging.keyinfo(f"{count} model complete. Still profiling... Save the intermediate results to {os.path.join(info_save_path, save_name)}.")

    # save information to json file
    save_profiled_results(models, os.path.join(info_save_path, save_name), detail, metrics)    
    logging.keyinfo(f"All {count} models complete. Save all success profiled results to {os.path.join(info_save_path, save_name)}.")

    return models

kernels = ["avgpool_block", "maxpool_block"]
for kernel_type in kernels:
    profile_models(
        backend=backend,
        kernel_type=kernel_type,
        models=f"{workspace}/predictor_build/results/{kernel_type}_prior.json",
        mode="predbuild",
        save_name=f"profiled_{kernel_type}.json"
    )
