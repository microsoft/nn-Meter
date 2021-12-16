# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import numpy as np
from sklearn.metrics import mean_squared_error


def get_accuracy(y_pred, y_true, threshold = 0.01):
    a = (y_true - y_pred) / y_true
    b = np.where(abs(a) <= threshold)
    c = abs(y_true - y_pred)
    return len(b[0]) / len(y_true)


def lat_metrics(y_pred, y_true):
    rmspe = (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))) * 100
    rmse = np.sqrt(mean_squared_error(y_pred, y_true))
    acc5 = get_accuracy(y_pred, y_true, threshold=0.05)
    acc10 = get_accuracy(y_pred, y_true, threshold=0.10)
    acc15 = get_accuracy(y_pred, y_true, threshold=0.15)
    return rmse, rmspe, rmse / np.mean(y_true), acc5, acc10, acc15


def get_flop(input_channel, output_channel, k, H, W, stride):
    paras = output_channel * (k * k * input_channel + 1)
    flops = 2 * H / stride * W / stride * paras
    return flops, paras


def get_conv_mem(input_channel, output_channel, k, H, W, stride):
    paras = output_channel * (k * k * input_channel + 1)
    mem = paras + output_channel * H / stride * W / stride + input_channel * H * W
    return mem


def get_depthwise_flop(input_channel, output_channel, k, H, W, stride):
    paras = output_channel * (k * k + 1)
    flops = 2 * H / stride * W / stride * paras
    return flops, paras


def get_flops_params(blocktype, hw, cin, cout, kernelsize, stride):
    if "dwconv" in blocktype:
        return get_depthwise_flop(cin, cout, kernelsize, hw, hw, stride)
    elif "conv" in blocktype:
        return get_flop(cin, cout, kernelsize, hw, hw, stride)
    elif "fc" in blocktype:
        flop = (2 * cin + 1) * cout
        return flop, flop


def read_kernel_latency(filename):
    f = open(filename, "r")
    X = []
    Y = []
    stds = []
    erros = []
    while True:
        line = f.readline()
        if not line:
            break
        content = line.split(",")
        if len(content) > 1 and content[-2] != "":
            op = content[1]
            features = [int(x) for x in content[2].split("_")]
            k = 1
            if len(features) == 5 and "concat" not in op:
                (hw, cin, cout, k, s) = features
            elif len(features) == 7:
                (hw, cin, cout, k, s, k1, s1) = features
                flops, params = get_flops_params("conv-bn-relu", hw, cin, cout, k, s)
                mem = get_conv_mem(cin, cout, k, hw, hw, s)
                features.append(flops / 2e6)
                features.append(params / 1e6)
                features.append(mem / 1e6)

            elif (
                len(features) == 2
                and "global" not in op
                and "hswish" not in op
                and "bnrelu" not in op
                and "relu" not in op
                and "bn" not in op
            ):
                (cin, cout) = features
                hw = 1
                k = 1
                s = 1
            elif len(features) == 2 and ("global" in op or "hswish" in op):
                (hw, cin) = features
                k = 1
            elif len(features) == 2 and op in ["bnrelu", "bn", "relu"]:
                (cin, hw) = features
                features = [hw, cin]
                k = 1
            elif len(features) == 3 and op == "addrelu":
                (hw, cin1, cin2) = features

            elif len(features) == 3 and ("shuffle" in op or "split" in op):
                (cin, hw, group) = features
                features1 = [hw, cin]
                features = features1
            elif len(features) == 3 and "se" in op:
                (hw, cin, group) = features
                features1 = [hw, cin]
                features = features1
            elif "concat" in op:
                hw = features[0]
                n = len(features[1:])
                features1 = [hw, n] + features[1:]
                if n < 4:
                    features1 += [0] * (4 - n)
                features = list(features1)
            else:
                (hw, cin, k, s) = features
                cout = cin
                features1 = [hw, cin, cout, k, s]
                features = features1

            if k < 9:
                latency = float(content[3])
                if latency > 0:  ##movidius: <1000
                    try:
                        std = float(content[4])
                        e = std / latency * 100
                    except:
                        std = 0
                        e = 0
                    if (
                        "pool" not in op
                        and "global" not in op
                        and "shuffle" not in op
                        and "split" not in op
                        and "se" not in op
                        and "hswish" not in op
                        and "concat" not in op
                        and op not in ["bnrelu", "addrelu", "bn", "relu"]
                    ):
                        flops, params = get_flops_params(op, hw, cin, cout, k, s)
                        features.append(flops / 2e6)
                        features.append(params / 1e6)
                        name = (
                            "_".join([str(hw), str(cin), str(cout), str(k), str(s)])
                            + ".tflite"
                        )

                    stds.append(std)
                    erros.append(std / latency * 100)
                    writeline = (
                        "_".join(str(x) for x in features)
                        + ","
                        + str(latency)
                        + ","
                        + str(std)
                        + ","
                        + str(std / latency * 100)
                    )
                    if features not in X:
                        flag1 = True
                        if op in "concat":
                            for fe in features:
                                if fe > 900:
                                    flag1 = False
                        if flag1:
                            X.append(features)
                            Y.append(latency)
    return X, Y


def get_feature(op, inputh, cin, cout, ks, s):
    if s != None and "conv" in op:

        flops, params = get_flops_params(op, inputh, cin, cout, ks, s)
        features = [inputh, cin, cout, ks, s, flops / 2e6, params / 1e6]

    elif "fc" in op:
        flop = (2 * cin + 1) * cout
        features = [cin, cout, flop / 2e6, flop / 1e6]

    elif "pool" in op:
        features = [inputh, cin, cout, ks, s]

    elif "se" in op:
        features = [inputh, cin]

    elif op in ["hwish", "hswish"]:
        features = [inputh, cin]

    return features
