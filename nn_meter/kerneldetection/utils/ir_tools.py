# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import copy
from .constants import OP_ALIAS


def convert_nodes(graph):
    """
    Resolve inconsistency between ONNX and Tensorflow
    """
    new_graph = copy.deepcopy(graph)

    for _, node in new_graph.items():
        type = node["attr"]["type"]
        new_type = OP_ALIAS.get(type, type)
        attr = node["attr"]["attr"]

        if "kernel_shape" in attr:
            attr["ks"] = attr["kernel_shape"]
            del attr["kernel_shape"]

        if "weight_shape" in attr and attr["weight_shape"] is not None:
            attr["ks"] = attr["weight_shape"][0:2]
            del attr["weight_shape"]

        if "ksize" in attr:
            attr["ks"] = attr["ksize"]
            del attr["ksize"]

        if new_type == "split" and "axis" in attr:
            attr["split_dim"] = attr["axis"]
            del attr["axis"]

        # workaround for add, mul, div, sub with const
        if new_type in ["add", "mul", "div", "sub"] and "input_shape" in node["attr"]:
            input_shape = node["attr"]["input_shape"]
            shape = input_shape[0] if input_shape[0] else input_shape[1]
            node["attr"]["input_shape"] = [shape] * len(input_shape)

        if new_type == "conv" and "group" in attr and "input_shape" in node["attr"]:
            group = attr["group"]
            cin = node["attr"]["input_shape"][0][3]
            if group == cin:
                new_type = "dwconv"

        node["attr"]["type"] = new_type

    return new_graph
