nni_type_map = {
    "aten::mul": "mul",
    "aten::floordiv": "div",
    "aten::reshape": "reshape",
    "aten::cat": "concat",
    "__torch__.torch.nn.modules.conv.Conv2d": "conv",
    "__torch__.torch.nn.modules.activation.ReLU": "relu",
    "__torch__.torch.nn.modules.batchnorm.BatchNorm2d": "bn",
    "__torch__.torch.nn.modules.linear.Linea": "fc",
    "__torch__.torch.nn.modules.pooling.AvgPool2d": "gap",
}


def int_to_list_modifier(attr):
    if isinstance(attr, int):
        return [attr, attr]


nni_attr_map = {
    "__all__": {
        "kernel_size": ("ks", int_to_list_modifier),
        "padding": ("pads", int_to_list_modifier),
        "stride": ("strides", int_to_list_modifier),
    },
    "concat": {
        "dim": ("axis", None),
    },
}
