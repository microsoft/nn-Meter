import tensorflow as tf 
from nn_meter.builder.utils import get_inputs_by_shapes, get_tensor_by_shapes
import numpy as np
from tensorflow import keras


config_for_blocks = {
      "conv_bn_relu":         ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
      "conv_block":           ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
      "conv_bn_relu_maxpool": ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES", "POOL_STRIDES"],
      "conv_bn_hswish":       ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
      "dwconv_bn_relu":       ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
      "dwconv_block":         ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
      "dwconv_bn_hswish":     ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
      "maxpool_block":        ["HW", "CIN", "KERNEL_SIZE", "POOL_STRIDES"],
      "avgpool_block":        ["HW", "CIN", "KERNEL_SIZE", "POOL_STRIDES"],
      "fc_block":             ["CIN", "COUT"],
      "hswish_block":         ["HW", "CIN"],
      "se_block":             ["HW", "CIN"],
      "global_avgpool_block": ["HW", "CIN"],
      "split_block":          ["HW", "CIN"],
      "channel_shuffle":      ["HW", "CIN"],
      "bn_relu":              ["HW", "CIN"],
      "concat_block":         ["HW", "CIN"],
      "concat_pad":           ["HW", "CIN"],
      "add_relu":             ["HW", "CIN"],
      "add_block":            ["HW", "CIN"],
      "bn_block":             ["HW", "CIN"],
      "relu_block":           ["HW", "CIN"]
}


if __name__ == '__main__':
    config = {}
    config["HW"] = 28
    config["CIN"] = 16
    config["COUT"] = 32
    config["KERNEL_SIZE"] = 3
    config["POOL_STRIDES"] = 2
    config["STRIDES"] = 1
    
    for block_type in config_for_blocks:
        print(f"######### {block_type} #########")
        from nn_meter.builder.nn_generator.predbuild_model import get_predbuild_model
        model, input_shape, needed_config = get_predbuild_model(block_type, config)
        output_shape = [mod.shape for mod in model(get_inputs_by_shapes([input_shape]))]
        
        # save model
        keras.models.save_model(model, "./testmodel")
        
        # import pdb; pdb.set_trace()
        model_path = "./testmodel"
        model = tf.keras.models.load_model(model_path)
        restore_shape = [mod.shape for mod in model(get_tensor_by_shapes([input_shape]))]
        
        # print(needed_config)
        print(f'input_shape: {input_shape}; output_shape: {output_shape}; model_restore_shape: {restore_shape}')
        assert output_shape == restore_shape
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        open("./testmodel.tflite", 'wb').write(tflite_model)
