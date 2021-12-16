# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from .networks import blocks
from nn_meter.builder.utils import get_inputs_by_shapes


def get_block_by_name(name, input_shape, config = None):
    block = getattr(blocks, name)(input_shape, config)
    return block


def get_predbuild_model(block_type, config):
      """ get the nn model for predictor build. returns: input_tensors, output_tensors, configuration_key, and graphname, they are for saving tensorflow v1.x models
      """
      config_idx = {
            "conv_bn_relu":         ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDE"],
            "conv":                 ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDE"],
            "conv-bn-relu-maxpool": ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDE"],
            "cconv-bn-hswish":      ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDE"],
            "dwconv-bn-relu":       ["HW", "CIN", "KERNEL_SIZE", "STRIDE"],
            "dwconv":               ["HW", "CIN", "KERNEL_SIZE", "STRIDE"],
            "dwconv-bn-hswish":     ["HW", "CIN", "KERNEL_SIZE", "STRIDE"],
            "maxpool":              ["HW", "CIN", "KERNEL_SIZE", "STRIDE"],
            "avgpool":              ["HW", "CIN", "KERNEL_SIZE", "STRIDE"],
            "fc":                   ["CIN", "COUT"],
            "hswish":               ["HW", "CIN"],
            "se":                   ["HW", "CIN"],
            "global-avgpool":       ["HW", "CIN"],
            "split":                ["HW", "CIN"],
            "channel-shuffle":      ["HW", "CIN"],
            "bn_relu":              ["HW", "CIN"],
            "concat":               ["HW", "CIN"],
            "concat_pad":           ["HW", "CIN"],
            "add_relu":             ["HW", "CIN"],
            "add":                  ["HW", "CIN"],
            "bn":                   ["HW", "CIN"],
            "relu":                 ["HW", "CIN"]
      }
      needed_config = config_idx[block_type]
      input_shape = [config["HW"], config["HW"], config["CIN"]]
      model = get_block_by_name(block_type, input_shape, config = None)
      model(get_inputs_by_shapes([input_shape]))
      return model, input_shape, config
