# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

config_for_kernel = {
    # conv
    "conv_bn_relu":         ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv_bn_relu6":        ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv_bn":              ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv_relu":            ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv_relu6":           ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv_hswish":          ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv_block":           ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv_bn_hswish":       ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    # dwconv
    "dwconv_bn":            ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
    "dwconv_relu":          ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
    "dwconv_relu6":         ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
    "dwconv_bn_relu":       ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
    "dwconv_bn_relu6":      ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
    "dwconv_block":         ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
    "dwconv_bn_hswish":     ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
    # others
    "maxpool_block":        ["HW", "CIN", "KERNEL_SIZE", "POOL_STRIDES"],
    "avgpool_block":        ["HW", "CIN", "KERNEL_SIZE", "POOL_STRIDES"],
    "fc_block":             ["CIN", "COUT"],
    "concat_block":         ["HW", "CIN1", "CIN2", "CIN3", "CIN4"],
    "split_block":          ["HW", "CIN"],
    "channel_shuffle":      ["HW", "CIN"],
    "se_block":             ["HW", "CIN"],
    "globalavgpool_block": ["HW", "CIN"],
    "bn_relu":              ["HW", "CIN"],
    "bn_block":             ["HW", "CIN"],
    "hswish_block":         ["HW", "CIN"],
    "relu_block":           ["HW", "CIN"],
    "add_relu":             ["HW", "CIN"],
    "add_block":            ["HW", "CIN"], 
}