# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from .utils import get_kernel_name
from .extract_feature import get_predict_features


# to speed up, we merge conv and dwconv related kernels into
def merge_conv_kernels(kernelname):
    if "conv" in kernelname and "dwconv" not in kernelname:
        return "conv-bn-relu"
    elif "dwconv" in kernelname:
        return "dwconv-bn-relu"
    else:
        return kernelname


def predict_model(model, predictors):
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
            pys = pred.predict(dicts[kernel])
            if len(pys) != 0:
                py += sum(pys)

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
