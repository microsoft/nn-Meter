# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import numpy as np
from sklearn.metrics import mean_squared_error


def get_kernel_name(optype):
    """
    for many similar kernels, we use one kernel predictor since their latency difference is negligible,
    return the kernel name via the optype

    """
    if "conv" in optype and "dwconv" not in optype:
        optype = "conv-bn-relu"
    if "dwconv" in optype:
        optype = "dwconv-bn-relu"
    if optype == "fc-relu":
        optype = "fc"
    if optype == "max-pool":
        optype = "maxpool"
    if optype == "avg-pool":
        optype = "avgpool"
    if optype in ["global-pool", "gap"]:
        optype = "global-avgpool"
    if optype == "channel_shuffle":
        optype = "channelshuffle"
    if optype in ["bn-relu"]:
        optype = "bnrelu"
    if optype in ["add-relu"]:
        optype = "addrelu"

    if optype in ["SE", "SE-relu", "se", "se-relu"]:
        optype = "se"

    return optype


def get_accuracy(y_pred, y_true, threshold=0.01):
    a = (y_true - y_pred) / y_true
    b = np.where(abs(a) <= threshold)
    return len(b[0]) / len(y_true)


def latency_metrics(y_pred, y_true):
    """
    evaluation metrics for prediction performance
    """
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    rmspe = (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))) * 100
    rmse = np.sqrt(mean_squared_error(y_pred, y_true))
    acc5 = get_accuracy(y_pred, y_true, threshold=0.05)
    acc10 = get_accuracy(y_pred, y_true, threshold=0.10)
    acc15 = get_accuracy(y_pred, y_true, threshold=0.15)
    return rmse, rmspe, rmse / np.mean(y_true), acc5, acc10, acc15