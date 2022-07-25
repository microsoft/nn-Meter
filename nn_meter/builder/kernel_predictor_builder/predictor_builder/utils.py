import json
import numpy as np
from sklearn.metrics import mean_squared_error


def get_accuracy(y_pred, y_true, threshold = 0.01):
    a = (y_true - y_pred) / y_true
    b = np.where(abs(a) <= threshold)
    return len(b[0]) / len(y_true)


def latency_metrics(y_pred, y_true):
    rmspe = (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))) * 100
    rmse = np.sqrt(mean_squared_error(y_pred, y_true))
    acc5 = get_accuracy(y_pred, y_true, threshold=0.05)
    acc10 = get_accuracy(y_pred, y_true, threshold=0.10)
    acc15 = get_accuracy(y_pred, y_true, threshold=0.15)
    return rmse, rmspe, rmse / np.mean(y_true), acc5, acc10, acc15


def get_conv_flop_params(hw, cin, cout, kernel_size, stride):
    params = cout * (kernel_size * kernel_size * cin + 1)
    flops = 2 * hw / stride * hw / stride * params
    return flops, params


def get_dwconv_flop_params(hw, cout, kernel_size, stride):
    params = cout * (kernel_size * kernel_size + 1)
    flops = 2 * hw / stride * hw / stride * params
    return flops, params


def get_fc_flop_params(cin, cout):
    params = (2 * cin + 1) * cout
    flops = params
    return flops, params


def get_flops_params(kernel_type, config):
    if "dwconv" in kernel_type:
        hw, cin, kernel_size, stride = config["HW"], config["CIN"], \
            config["KERNEL_SIZE"], config["STRIDES"]
        return get_dwconv_flop_params(hw, cin, kernel_size, stride)
    elif "conv" in kernel_type:
        hw, cin, cout, kernel_size, stride = config["HW"], config["CIN"], \
            config["COUT"], config["KERNEL_SIZE"], config["STRIDES"]
        return get_conv_flop_params(hw, cin, cout, kernel_size, stride)
    elif "fc" in kernel_type:
        cin, cout = config["CIN"], config["COUT"]
        return get_fc_flop_params(cin, cout)


def collect_kernel_data(kernel_data, predict_label = 'latency'):
    if isinstance(kernel_data, dict):
        return kernel_data

    config, label = kernel_data
    if isinstance(config, list):
        config = collect_data(config)
    else:
        with open(config, 'r') as fp:
            config = json.load(fp)

    if isinstance(label, list):
        label = collect_data(label)
    else:
        with open(label, 'r') as fp:
            label = json.load(fp)
    if predict_label == 'latency':
        from nn_meter.builder.backend_meta.utils import read_profiled_results
        label = read_profiled_results(label)

    for modules in config.keys():
        for model_id in config[modules].keys():
            try:
                config[modules][model_id][predict_label] = label[modules][model_id][predict_label]
            except:
                pass

    return config


def collect_data(file_list):
    file_list_copy = file_list[:]

    from ...utils import merge_info
    data = file_list_copy.pop(0)
    with open(data, 'r') as fp:
        data = json.load(fp)
    for file in file_list_copy:
        data = merge_info(new_info=file, prev_info=data)
    return data
