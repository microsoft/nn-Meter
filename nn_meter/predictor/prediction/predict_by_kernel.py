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
    # res = ''
    for kernel in dicts:
        kernelname = get_kernel_name(kernel)
        if kernelname in predictors:
            pred = predictors[kernelname]
            pys = pred.predict(dicts[kernel]) # in unit of ms
            if len(pys) != 0:
                py += sum(pys)

    #         for lat, feature in zip(pys, dicts[kernel]):
    #             res += f'{kernel}, {lat}, {", ".join(list(map(str, feature)))}\n'
    # open("/data/jiahang/working/nn-Meter/examples/test_quantize_latency_predictor/predict_res.txt", 'a').write(res + 'end\n')  

    return py


def nn_predict(predictors, kernel_units):
    """
    @params:
    predictors: dictionary object, key: kernel name, object: loaded pkl latency model
    kernel_units: the divided kernel units and the features of a model.
    """

    features = get_predict_features(kernel_units)
    py = predict_model(features, predictors)
    return py
