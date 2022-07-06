# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from .utils import get_kernel_name
from .extract_feature import get_predict_features


def merge_conv_kernels(kernelname):
    """
    to speed up, we merge conv and dwconv related kernels into one kernel by their name
    """
    if "conv" in kernelname and "dwconv" not in kernelname:
        return "conv-bn-relu"
    elif "dwconv" in kernelname:
        return "dwconv-bn-relu"
    else:
        return kernelname


def predict_model(model, predictors):
    """
    @params:
    model: the model config with prediction features
    predictors: loaded pkl predictors
    """
    py = 0
    dicts = {}
    for layer in model:
        kernel = list(model[layer].keys())[0]
        features = model[layer][kernel]
        rkernel = merge_conv_kernels(kernel)
        if rkernel not in dicts:
            dicts[rkernel] = []
        dicts[rkernel].append(features)

    for kernel in dicts:
        kernelname = get_kernel_name(kernel)
        if kernelname in predictors:
            pred = predictors[kernelname]
            pys = pred.predict(dicts[kernel]) # in unit of ms
            if len(pys) != 0:
                py += sum(pys)

    return py


def detailed_predict_model(model, predictors):
    """
    @params:
    model: the model config with prediction features
    predictors: loaded pkl predictors
    """
    py = 0
    dicts = {}
    layer_idx = 0
    layer_map = {}
    block_latency = []
    for layer in model:
        kernel = list(model[layer].keys())[0]
        features = model[layer][kernel]
        rkernel = merge_conv_kernels(kernel)
        kernelname = get_kernel_name(rkernel)
        if kernelname not in dicts:
            dicts[kernelname] = []
        dicts[kernelname].append(features)
        layer_map[layer_idx] = [kernelname, len(dicts[kernelname])-1] # kerne_type, idx
        layer_idx += 1

    # predict latency
    block_latency = {}
    for kernelname in dicts:
        if kernelname in predictors:
            pred = predictors[kernelname]
            pys = pred.predict(dicts[kernelname]) # in unit of ms
            if len(pys) != 0:
                block_latency[kernelname] = pys
                py += sum(pys)

    # merge
    layer_idx = 0
    block_result = []
    op_no = 1
    for layer in model:
        kernelname = list(model[layer].keys())[0]
        
        rkernelname = get_kernel_name(merge_conv_kernels(kernelname))
        if rkernelname in predictors:
            curr_type, curr_idx = layer_map[layer_idx]
            curr_lat = block_latency[curr_type][curr_idx]
            block_result.append([op_no, rkernelname, model[layer][kernelname], curr_lat, layer.split("#")[0]])
            op_no += 1
        else:
            curr_type, curr_idx = layer_map[layer_idx]
            curr_lat = 0.0
            block_result.append([op_no, rkernelname, model[layer][kernelname], curr_lat, layer.split("#")[0]])
            op_no += 1
        layer_idx += 1

    return py, block_result


def nn_predict(predictors, kernel_units):
    """
    @params:
    predictors: dictionary object, key: kernel name, object: loaded pkl latency model
    kernel_units: the divided kernel units and the features of a model.
    """

    features = get_predict_features(kernel_units)
    py = predict_model(features, predictors)
    return py


def detailed_nn_predict(predictors, kernel_units):
    """
    @params:
    predictors: dictionary object, key: kernel name, object: loaded pkl latency model
    kernel_units: the divided kernel units and the features of a model.
    """

    features = get_predict_features(kernel_units)
    py = detailed_predict_model(features, predictors)
    return py
