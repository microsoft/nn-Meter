# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
from .networks import blocks
from .utils import save_model
from nn_meter.builder.utils import get_inputs_by_shapes


config_for_blocks = {
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
    "concat_block":         ["HW", "NS", "CINS"],
    "split_block":          ["HW", "CIN"],
    "channel_shuffle":      ["HW", "CIN"],
    "se_block":             ["HW", "CIN"],
    "global_avgpool_block": ["HW", "CIN"],
    "bn_relu":              ["HW", "CIN"],
    "bn_block":             ["HW", "CIN"],
    "hswish_block":         ["HW", "CIN"],
    "relu_block":           ["HW", "CIN"],
    "add_relu":             ["HW", "CIN"],
    "add_block":            ["HW", "CIN"],
}


def get_block_by_name(name, input_shape, config = None):
    block = getattr(blocks, name)(input_shape, config)
    return block


def get_predbuild_model(block_type, config, savepath=None):
      """ get the nn model for predictor build. returns: input_tensors, output_tensors, configuration_key, and graphname, they are for saving tensorflow v1.x models
      """
      try:
            needed_config = {k: config[k] for k in config_for_blocks[block_type]}
            if "POOL_STRIDES" in config_for_blocks[block_type] and "POOL_STRIDES" not in config:
                  needed_config["POOL_STRIDES"] = config["STRIDES"]
      except:
            raise NotImplementedError("The block_type you called is not exist in our model zoo. Please implement the block and try again.")
      if block_type == "fc_block":
            input_shape = [1, config["CIN"]]
            input_tensor_shape = [input_shape]
      elif block_type == "concat_block":
            input_shape = [[config["HW"], config["HW"], config["CINS"][i]] for i in range(config["NS"])]
            input_tensor_shape = input_shape
      else:
            input_shape = [config["HW"], config["HW"], config["CIN"]]
            input_tensor_shape = [input_shape]
      model = get_block_by_name(block_type, input_shape, needed_config)
      model(get_inputs_by_shapes(input_tensor_shape))
      if savepath:
            save_model(model, savepath)
            logging.info(f"{block_type} model is generated and saved to {savepath}.")
      else:
            logging.info(f"{block_type} model is generated.")
      return model, input_shape, needed_config
