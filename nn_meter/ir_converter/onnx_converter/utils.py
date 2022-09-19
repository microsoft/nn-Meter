# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
def get_tensor_shape(tensor):
    shape = []
    for dim in tensor.type.tensor_type.shape.dim:
        shape.append(dim.dim_value)
    if len(shape) == 4:
        shape = [shape[0], shape[2], shape[3], shape[1]]
    return shape
