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


def get_config_by_features(kernel_type, feature):
    from ..config_lib import config_for_kernel
    config_name = config_for_kernel[kernel_type]

    # remove flops and params num feature from feature vector
    if "conv" in kernel_type or "dwconv" in kernel_type or "fc" in kernel_type:
        feature = feature[:-2]

    assert len(config_name) == len(feature)
    config = {k: v for k, v in zip(config_name, feature)}
    return config 