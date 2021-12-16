import tensorflow as tf 
from nn_meter.builder.utils import get_inputs_by_shapes, get_tensor_by_shapes
import numpy as np
from tensorflow import keras


def conv(input_shape, config = None):
    output_shape = [shape for shape in input_shape[:2]] + [config['cout']]
    return keras.layers.Conv2D(
            config['cout'],
            kernel_size=config['kernel_size'],
            padding=config['padding'],
            strides=config['strides']
        ), output_shape
    
    
def batch_norm(input_shape, config = None):
    return keras.layers.BatchNormalization(), input_shape

def relu(input_shape, config = None):
    return keras.layers.ReLU(), input_shape


def conv_bn_relu(input_shape, config):
    conv_op, out_shape = conv(input_shape, config)
    bn_op, out_shape = batch_norm(out_shape, config)
    relu_op, _ = relu(out_shape, config)
    
    class ConvBnRelu(tf.keras.Model):
        def __init__(self, conv_op, bn_op, relu_op):
            super().__init__()
            self.conv = conv_op
            self.bn = bn_op
            self.relu = relu_op

        def call(self, inputs):
            x = self.conv(inputs)
            x = self.bn(x)
            x = self.relu(x)
            return x

    return ConvBnRelu(conv_op, bn_op, relu_op)


if __name__ == '__main__':
    hw_in = 28
    config = {}
    config["cin"] = 16
    config["cout"] = 64
    config["kernel_size"] = 3
    config["strides"] = 1
    config["padding"] = 'same'
    input_shape = [hw_in, hw_in, config["cin"]]
    
    import pdb; pdb.set_trace()
    
    model = conv_bn_relu(input_shape, config)
    model(get_inputs_by_shapes([input_shape]))
    
    # save model
    keras.models.save_model(model, "./testmodel")
    
    
    model_path = "./testmodel"
    model = tf.keras.models.load_model(model_path)
    model(get_tensor_by_shapes([input_shape]))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    