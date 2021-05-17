from .constants import *


def get_tensor_shape(tensor):
    shape = []
    for dim in tensor.type.tensor_type.shape.dim:
        shape.append(dim.dim_value)
    if len(shape) == 4:
        shape = [shape[0], shape[2], shape[3], shape[1]]
    return shape


def convert_attr(attr, type):
    def is_type(type, ts):
        if ts is None:
            return False
        elif ts == '__all__':
            return True
        else:
            return type in ts

    new_attr = {}

    for name, value in attr.items():
        new_name, ts = ATTR_ALIAS.get(name, (name, None))
        if is_type(type, ts):
            new_attr[new_name] = value
        else:
            new_attr[name] = value

    return new_attr
