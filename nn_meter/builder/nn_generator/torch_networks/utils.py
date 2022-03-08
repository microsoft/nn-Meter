# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn as nn

def get_tensor_by_shapes(shapes):
    if len(shapes) == 1:
        return torch.randn(size=[1] + shapes[0])
    else:
        return [torch.randn(size=[1] + shape) for shape in shapes]


def get_inputs_by_shapes(shapes):
    return get_tensor_by_shapes(shapes)
