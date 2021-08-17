# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
nni_type_map = {
    "aten::mul": "mul",
    "aten::floordiv": "div",
    "aten::reshape": "reshape",
    "aten::cat": "concat",
    "aten::split": "split",
    "__torch__.torch.nn.modules.conv.Conv2d": "conv",
    "__torch__.torch.nn.modules.activation.ReLU": "relu",
    "__torch__.torch.nn.modules.batchnorm.BatchNorm2d": "bn",
    "__torch__.torch.nn.modules.linear.Linear": "fc",
    "__torch__.torch.nn.modules.pooling.AvgPool2d": "gap",
    "__torch__.torch.nn.modules.pooling.MaxPool2d": "maxpool",
    "__torch__.torch.nn.modules.activation.Sigmoid": "sigmoid",
    "__torch__.torch.nn.modules.activation.Hardsigmoid": "hardsigmoid",
    "__torch__.torch.nn.modules.activation.Hardswish": "hswish",
    "__torch__.torch.nn.modules.activation.ReLU6": "relu",
    "__torch__.torch.nn.modules.activation.Softmax": "softmax",
}


def int_to_list_modifier(attr):
    if isinstance(attr, int):
        return [attr, attr]
    else:
        return list(attr)


nni_attr_map = {
    "__all__": {
        "kernel_size": ("ks", int_to_list_modifier),
        "padding": ("pads", int_to_list_modifier),
        "stride": ("strides", int_to_list_modifier),
        "groups": ("group", None),
    },
    "concat": {
        "dim": ("axis", None),
    },
}
