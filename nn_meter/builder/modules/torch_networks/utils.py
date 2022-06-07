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


def get_padding(ks, s, hw):
    """ choose padding value to make sure:
    if s = 1, out_hw = in_hw;
    if s = 2, out_hw = in_hw // 2;
    if s = 4, out_hw = in_hw // 4;
    """
    if hw % s == 0:
        pad = max(ks - s, 0)
    else:
        pad = max(ks - (hw % s), 0)
    if pad % 2 == 0:
        return pad // 2
    else:
        return pad // 2 + 1
