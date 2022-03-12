# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn as nn

def get_tensor_by_shapes(shapes, batch_size = 1):
    if len(shapes) == 1:
        return torch.randn(size=[batch_size] + shapes[0])
    else:
        return [torch.randn(size=[batch_size] + shape) for shape in shapes]


def get_inputs_by_shapes(shapes, batch_size = 1):
    return get_tensor_by_shapes(shapes, batch_size)
