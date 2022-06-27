# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import json
import time
import logging
from . import builder_config
from .utils import save_profiled_results, merge_info
from nn_meter.builder.backends import connect_backend
logging = logging.getLogger("nn-Meter")


def convert_models(backend, models, mode = 'predbuild', broken_point_mode = False):
    """ convert the model to the needed format by backend, in order to increase efficiency when profiling on device.

    @params:

    backend (subclass instance of BaseBackend): applied backend instance

    models (str or dict): the Dict of models or the path of the json file about models information 

    mode (str): the mode for running models, including ['ruletest', 'predbuild']

    broken_point_mode (boolean): broken_point_mode will skip all models have attributes "converted_model"

    """
    if isinstance(models, str):
        save_name = os.path.basename(models)
        with open(models, 'r') as fp:
            models = json.load(fp)
    else:
        save_name = "converted_results.json"

    workspace_path = builder_config.get('WORKSPACE', mode)
    model_save_path = os.path.join(workspace_path, 'testcases' if mode == 'ruletest' else 'kernels')
    os.makedirs(model_save_path, exist_ok=True)
    res_save_path = os.path.join(workspace_path, "results")
    os.makedirs(res_save_path, exist_ok=True)

    # convert models
    count = 0
    info_save_path = os.path.join(res_save_path, save_name)
    error_save_path = os.path.join(res_save_path, "convert_error.log")
    for module in models.values():
        for id, model in module.items():
            if broken_point_mode and 'converted_model' in model:
                continue
            try:
                model_path = model['model']
                converted_model = backend.convert_model(model_path, model_save_path, model['shapes'])
                model['converted_model'] = converted_model
                count += 1
            except Exception as e:
                open(error_save_path, 'a').write(f"{id}: {e}\n")

            # save information to json file for per 50 models
            if count % 50 == 0:
                with open(info_save_path, 'w') as fp:
                    json.dump(models, fp, indent=4)
                logging.keyinfo(f"{count} models complete. Still converting... Save the intermediate results to {info_save_path} ")

    # save information to json file
    with open(info_save_path, 'w') as fp:
        json.dump(models, fp, indent=4)
    logging.keyinfo(f"Complete converting all {count} models. Save the results to {info_save_path} " \
                    f"Failed information are saved in {error_save_path} (if any)")

    return models


def profile_models(backend, models, mode = 'ruletest', metrics = ["latency"], save_name = "profiled_results.json",
                   have_converted = False, log_frequency = 50, broken_point_mode = False, **kwargs):
    """ run models with given backend and return latency of testcase models

    @params:

    backend (subclass instance of BaseBackend): applied backend instance

    models (str or dict): the Dict of models or the path of the json file about models information 

    mode (str): the mode for running models, including ['ruletest', 'predbuild']

    metrics (list): required metrics to report. We only support latency for metric by now.

    save_name (str): the save name to store profiled results. The whole path should be `<workspace>/<mode-folder>/results/<save-name>`

    have_converted (boolean): if the model have been converted to the needed format by backend, the model will not be converted
        before profiling. The model path of `model['converted_model']` will be profiled on device directly. The conversion of
        model could be done by appling `nn_meter.builder.convert_models`

    broken_point_mode (boolean): broken_point_mode will check file in `<workspace>/<mode-folder>/results/<save-name>` (if the file exists)
        and skip all models already have attributes "latency"

    **kwargs: arguments for profiler, such as `taskset` and `close_xnnpack` in TFLite profiler
    """
    if isinstance(models, str):
        with open(models, 'r') as fp:
            models = json.load(fp)

    workspace_path = builder_config.get('WORKSPACE', mode)
    model_save_path = os.path.join(workspace_path, 'testcases' if mode == 'ruletest' else 'kernels')
    os.makedirs(model_save_path, exist_ok=True)
    res_save_path = os.path.join(workspace_path, "results")
    os.makedirs(res_save_path, exist_ok=True)
    info_save_path = os.path.join(res_save_path, save_name)

    # in broken point model, if the output file `<workspace>/<mode-folder>/results/<save-name>` exists,
    # load the existing latency and skip these model in profiling
    if broken_point_mode and os.path.isfile(info_save_path):
        from nn_meter.builder.backend_meta.utils import read_profiled_results
        with open(info_save_path, 'r') as fp:
            profiled_models = read_profiled_results(json.load(fp))
        for module_key, module in models.items():
            if module_key not in profiled_models:
                continue
            for id, model in module.items():
                if id in profiled_models[module_key]:
                    model.update(profiled_models[module_key][id])

    # profile models and get metric results
    count = 0    
    error_save_path = os.path.join(res_save_path, "profile_error.log")
    detail = builder_config.get('DETAIL', mode)
    save_name = save_name or "profiled_results.json"
    logging.info("Profiling ...")
    for module in models.values():
        for id, model in module.items():
            if broken_point_mode and 'latency' in model and model['latency'].avg != 0:
                continue
            if have_converted: # the models have been converted for the backend
                try:
                    model_path = model['converted_model']
                    profiled_res = backend.profile(model_path, metrics, model['shapes'], **kwargs)
                    for metric in metrics:
                        model[metric] = profiled_res[metric]
                    time.sleep(0.2)
                    count += 1
                except Exception as e:
                    open(error_save_path, 'a').write(f"{id}: {e}\n")
            else: # the models have not been converted
                try:
                    model_path = model['model']
                    profiled_res = backend.profile_model_file(model_path, model_save_path, model['shapes'], metrics, **kwargs)
                    for metric in metrics:
                        model[metric] = profiled_res[metric]
                    time.sleep(0.2)
                    count += 1
                except Exception as e:
                    open(error_save_path, 'a').write(f"{id}: {e}\n")

            # save information to json file for per 50 models
            if count > 0 and count % log_frequency == 0:
                save_profiled_results(models, info_save_path, detail, metrics)
                logging.keyinfo(f"{count} models complete. Still profiling... Save the intermediate results to {info_save_path} ")

    # save information to json file
    save_profiled_results(models, info_save_path, detail, metrics)    
    logging.keyinfo(f"All {count} models profiling complete. Save all success profiled results to {info_save_path} " \
                    f"Failed information are saved in {error_save_path} (if any)")

    return models


def sample_and_profile_kernel_data(kernel_type, sample_num, backend, sampling_mode = 'prior', configs = None, mark = '', detail = True,
                                   metrics = ["latency"], **kwargs):
    ''' sample kernel configs and profile kernel model based on configs
    '''
    from nn_meter.builder.kernel_predictor_builder import generate_config_sample

    # sample configs for kernel and generate models
    models = generate_config_sample(kernel_type, sample_num, mark=mark, 
                                     sampling_mode=sampling_mode, configs=configs)

    # connect to backend, run models and get latency
    backend = connect_backend(backend_name=backend)
    profiled_results = profile_models(backend, models, mode='predbuild', metrics=metrics, save_name=f"profiled_{kernel_type}.json")
    return profiled_results


def build_predictor_for_kernel(kernel_type, backend, init_sample_num = 1000, finegrained_sample_num = 10,
                               iteration = 5, error_threshold = 0.1, predict_label = "latency", mark = ""):
    """ 
    Build latency predictor for given kernel. This method contains three main steps:
    1. sample kernel configs and profile kernel model based on configs;
    2. initialize latency predictor of kernel based on the profiled data;
    3. adopt adaptive sampler with iteratively doing step 1 for finegrained sampling to improve predictor performance

    @params
    
    kernel_type (str): the type of kernel
    
    backend (str): the name of backend instance to profile models
    
    init_sample_num (int, optional): the data size for predictor initialization. Defaults to 1000.
    
    finegrained_sample_num (int, optional): the data size for adaptive sampling. For each data with error higher than 
        error_threshold, number of `finegrained_sample_num` data will be generated based the the large error data. Defaults to 10.

    iteration (int, optional): the iteration for sampling and training. Predictor training based on initial sampling is regarded as
    iteration 1, thus `iteration == 2` means one iteration for adaptive sampling. Defaults to 5.

    error_threshold (float, optional): the threshold of large error. Defaults to 0.1.

    predict_label (str): the predicting label to build kernel predictor. Defaults to "latency"
 
    """
    from nn_meter.builder.kernel_predictor_builder import build_predictor_by_data
    workspace_path = builder_config.get('WORKSPACE', 'predbuild')
    mark = mark if mark == "" else "_" + mark

    # init predictor builder with prior data sampler
    kernel_data = sample_and_profile_kernel_data(kernel_type, init_sample_num, backend, sampling_mode='prior', mark=f'prior{mark}')

    # use current sampled data to build regression model, and locate data with large errors in testset
    predictor, acc10, error_configs = build_predictor_by_data(kernel_type, kernel_data, backend, error_threshold=error_threshold, mark=f'prior{mark}',
                                                              save_path=os.path.join(workspace_path, "results"), predict_label=predict_label)
    logging.keyinfo(f'Iteration 0: acc10 {acc10}, error_configs number: {len(error_configs)}')

    for i in range(1, iteration):
        # finegrained sampling and profiling for large error data
        new_kernel_data = sample_and_profile_kernel_data(kernel_type, finegrained_sample_num, backend,
                                                         sampling_mode='finegrained', configs=error_configs, mark=f'finegrained{i}{mark}')

        # merge finegrained data with previous data and build new regression model
        kernel_data = merge_info(new_info=new_kernel_data, prev_info=kernel_data)
        predictor, acc10, error_configs = build_predictor_by_data(kernel_type, kernel_data, backend, error_threshold=error_threshold, mark=f'finegrained{i}{mark}',
                                                                  save_path=os.path.join(workspace_path, "results"), predict_label=predict_label)
        logging.keyinfo(f'Iteration {i}: acc10 {acc10}, error_configs number: {len(error_configs)}')

    return predictor, kernel_data


def build_initial_predictor_by_data(kernel_type, backend = None, init_sample_num = 20, error_threshold = 0.1, mark = '', predict_label = "latency"):
    return build_predictor_for_kernel(kernel_type, backend, init_sample_num=init_sample_num, iteration=1, error_threshold=error_threshold, predict_label=predict_label, mark=mark)


def build_adaptive_predictor_by_data(kernel_type, kernel_data, backend = None, finegrained_sample_num = 20, error_threshold = 0.1, mark = '', predict_label = "latency"):
    """ Run adaptive sampler in one iteration based 
    """
    workspace_path = builder_config.get('WORKSPACE', 'predbuild')
    save_path = os.path.join(workspace_path, "results")
    mark = 'finegrained' if mark == "" else f"finegrained_{mark}"

    from nn_meter.builder.kernel_predictor_builder import build_predictor_by_data, collect_kernel_data
    _, _, error_configs = build_predictor_by_data(kernel_type, kernel_data, backend = backend, error_threshold=error_threshold, save_path=None, predict_label=predict_label)
    new_kernel_data = sample_and_profile_kernel_data(kernel_type, finegrained_sample_num, backend,
                                                     sampling_mode='finegrained', configs=error_configs, mark=mark)

    # merge finegrained data with previous data and build new regression model
    kernel_data = merge_info(new_info=new_kernel_data, prev_info=collect_kernel_data(kernel_data))
    predictor, acc10, error_configs = build_predictor_by_data(kernel_type, kernel_data, backend, error_threshold=error_threshold,
                                                              mark=mark, save_path=save_path, predict_label=predict_label)
    logging.keyinfo(f'{mark}: acc10 {acc10}, error_configs number: {len(error_configs)}')
    return predictor, kernel_data


def build_latency_predictor(backend):
    """ 
    Build latency predictor for all kernel in `<workspace-path>/configs/predictorbuild_config.yaml`

    @params

    backend (str): the name of backend instance to profile models

    """
    kernels = builder_config.get("KERNELS", 'predbuild')

    for kernel_type in kernels:
        init_sample_num = kernels[kernel_type]["INIT_SAMPLE_NUM"]
        finegrained_sample_num = kernels[kernel_type]["FINEGRAINED_SAMPLE_NUM"]
        iteration = kernels[kernel_type]["ITERATION"]
        error_threshold = kernels[kernel_type]["ERROR_THRESHOLD"]
        build_predictor_for_kernel(
            kernel_type, backend, 
            init_sample_num = init_sample_num,
            finegrained_sample_num = finegrained_sample_num,
            iteration = iteration,
            error_threshold = error_threshold
            )
