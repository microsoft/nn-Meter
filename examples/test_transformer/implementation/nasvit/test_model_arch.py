
import tensorflow as tf
import random
import shutil
import os
import math
import pickle
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    drop_block,
    layer_norm,
    se_module,
    
    MultiHeadRelativePositionalEmbedding,
)
from tensorflow.keras import layers

from tensorflow.keras.layers import BatchNormalization
from keras_cv_attention_models.common_layers import hard_swish
import multiprocessing
# from tflite import ADB, PIXEL4_SERIAL_NUMBER

from keras_cv_attention_models.levit.levit import mhsa_with_multi_head_position_windows, res_mlp_block, mhsa_with_multi_head_position_windows_layer_norm, res_mlp_block_layer_norm
from keras_cv_attention_models.attention_layers import depthwise_conv2d_no_bias, batchnorm_with_activation, conv2d_no_bias, activation_by_name,layer_norm

# ACT = 'swish'
ACT = 'hard_swish'
LAYER_NORM = False
DOWNSAMPLING = [True, True, True, False, True, True]
TALKING_HEAD = False

def dsconv(inputs, channel, act, strides, kernel_size, exp, use_se=False):
    inp_channel = inputs.shape[-1]
    print('res conv', inputs.shape)
    
    nn = conv2d_no_bias(inputs, inp_channel * exp, 1, strides=1, padding="SAME")
    nn = batchnorm_with_activation(nn, activation=act)
    nn = depthwise_conv2d_no_bias(inputs=nn, kernel_size=kernel_size, strides=strides, padding="SAME")

    nn = batchnorm_with_activation(nn, activation=act)

    if use_se:
        # layers.GlobalAveragePooling2D(keepdims=True)
        se_d = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(nn)
        # print(se_d.shape)
        # se_d = tf.expand_dims(se_d, axis=1)
        # se_d = tf.expand_dims(se_d, axis=1)
        se_d = conv2d_no_bias(se_d, inp_channel * exp //4, 1, strides=1, use_bias=True)
        se_d = tf.nn.relu(se_d)
        se_d = conv2d_no_bias(se_d, inp_channel * exp, 1, strides=1, use_bias=True)
        se_d = tf.nn.relu6(se_d + 3) / 6
        nn = se_d * nn

    nn = conv2d_no_bias(nn, channel, 1, strides=1, padding="SAME")
    nn = batchnorm_with_activation(nn, activation=None)
    if strides == 1 and inp_channel == channel:
        nn = nn + inputs
    
    # if stride == 1 and inp_channel == channel:
    #     ch = conv2d_no_bias()
    return nn


def first_conv(inputs, channels, act):
    nn = conv2d_no_bias(inputs, channels, 3, strides=2, padding='SAME', use_bias=False)
    nn = batchnorm_with_activation(nn, activation=act)
    nn = depthwise_conv2d_no_bias(nn, kernel_size=3, strides=1, padding='SAME',use_bias=False)
    nn = batchnorm_with_activation(nn, activation=act)
    nn = conv2d_no_bias(nn, channels, 1, strides=1, padding='SAME', use_bias=False)
    return nn


def attention_downsampling(inputs, out_channels, downsampling, dwconv=False, exp=6, use_se=False):
    B, L, C = inputs.shape
    # print(inputs.shape)
    H = W = int(L**0.5)
    x = tf.reshape(inputs, [-1, H, W, C])
    
    if not dwconv:
        if downsampling:
            nn = conv2d_no_bias(x, out_channels, 3, strides=2, padding='SAME', use_bias=False)
        else:
            nn = conv2d_no_bias(x, out_channels, 1, strides=1, padding='SAME', use_bias=False)
        nn = batchnorm_with_activation(nn, activation=ACT)
    else:
        print(x.shape)
        if downsampling:
            nn = dsconv(x, out_channels, ACT, 2, 3, exp, use_se=use_se)
        else:
            nn = dsconv(x, out_channels, ACT, 1, 3, exp, use_se=use_se)
        print(nn.shape)
        print('-'*30)
    B, H, W, C = nn.shape
    print('res trans', H)
    nn = tf.reshape(nn, [-1, int(H*W), out_channels])

    return nn


def build_model(inputs, channels, depths, conv_ratio, kr_size, mlp_ratio, num_heads, window_size, qk_scale, v_scale, nasvit_arch=True, ds=DOWNSAMPLING, se=None, stage=['C', 'C', 'C', 'T', 'T', 'T'], num_mlp=1, reproduce_nasvit=False):
    return build_model_new(inputs, channels, depths, conv_ratio, kr_size, mlp_ratio, num_heads, window_size, qk_scale, v_scale, downsampling=ds, nasvit_arch=nasvit_arch, se=se, stage=stage, num_mlp=num_mlp, reproduce_nasvit=reproduce_nasvit)


def build_model_new(inputs, channels, depths, conv_ratio, kr_size, mlp_ratio, num_heads, window_size, qk_scale, v_scale, downsampling, nasvit_arch=False, se=None, stage=['C', 'C', 'C', 'T', 'T', 'T'], num_mlp=1, reproduce_nasvit=False):
    # # swish
    # op_name = "swish"
    # nn = activation_by_name(inputs, activation=ACT, name='act')

    # layer norm
    # op_name = "ln"
    # nn = layer_norm(inputs, name='ln_attn')
    
    # batch norm
    # op_name = "bn"
    # nn = batchnorm_with_activation(inputs, activation=None, name="bn_attn")
    
    # # bn attention
    # op_name = "bn_attention"
    # nn = mhsa_with_multi_head_position_windows(inputs, 64, 8, 8, 32, 1, nasvit_arch=True, name=f'layer')
    
    # # ln attention
    # op_name = "ln_attention"
    # nn = mhsa_with_multi_head_position_windows_layer_norm(inputs, 64, 8, 8, 32, 1, nasvit_arch=True, name=f'layer')
    
    # # bn mlp
    # op_name = "bn_mlp"
    # nn = res_mlp_block(inputs, 1, name='ffn')
    
    # # ln mlp
    # op_name = "ln_mlp_exp1_repeat"
    # nn = res_mlp_block_layer_norm(inputs, 1, name='ffn', activation='swish')
    # nn = res_mlp_block_layer_norm(nn, 1, name='ffn_2', activation='swish')
    
    # op_name = "ln_mlp_exp2_norepeat"
    # nn = res_mlp_block_layer_norm(inputs, 2, name='ffn', activation='swish')
    
    # linear
    op_name = "linear"
    nn = tf.keras.layers.Activation("linear", name="output")(nn)  # Identity, Just need a name here
    
    
    model = tf.keras.models.Model(inputs, nn)
    
    return model, op_name
                

def tf2tflite(saved_model_path: str, output_path: str, is_keras=False, quantization='None', use_flex=True, input_shape=None):
    assert output_path.endswith('.tflite')
    if os.path.dirname(output_path) != '' and not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    if is_keras:
        model = tf.keras.models.load_model(saved_model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path) # path to the SavedModel directory
    
    if quantization == 'float16':
        print('Apply float16 quantization.')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == 'dynamic':
        print('Apply dynamic range quantization.')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == 'int8':
        print('Apply int8 quantization')
        def representative_data_gen():
            #  for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
            #      yield [input_value]
            for _ in range (100):
                yield [tf.random.normal(input_shape)]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # type: ignore
        converter.representative_dataset = representative_data_gen  # type: ignore
        # Ensure that if any ops can't be quantized, the converter throws an error
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Set the input and output tensors to uint8 (APIs added in r2.3)
        # Change to int8 on 2021/11/19
        converter.experimental_new_converter = True
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    if use_flex:
        print('Use Flex Delegate.')
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8 if quantization== 'int8' else tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        ]
    else:
        print('Not use Flex Delegate.')

    tflite_model = converter.convert()

    # Save the model.
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    del tflite_model
    print(f'Successfully convert tflite model to {output_path}.')



def export_op():
    from config import NASVIT_SMALLEST, NASVIT_A1_L, NASVIT_A2_L, NASVIT_A3_L, NASVIT_A4_L
    STAGE = ['C', 'C', 'T', 'T', 'T', 'T']
    
    for a, name in zip([NASVIT_SMALLEST], ['a0']):
        res, channels, depths, conv_ratio, kr_size, mlp_ratio, num_heads, window_size, qk_scale, v_scale, use_se = a
        # inputs = tf.keras.layers.Input((res, res, 3))
        inputs = tf.keras.layers.Input((144, 64))
        ds = [True, True, True, True, False, True]
        model, op_name = build_model(inputs, channels, depths, conv_ratio, kr_size, mlp_ratio, num_heads, window_size, qk_scale, v_scale, nasvit_arch=True, ds=ds, se=use_se, stage=STAGE, num_mlp=2, reproduce_nasvit=True)
        # model(tf.random.normal((1, res, res, 3)))
        model(tf.random.normal((1, 144, 64)))
        tf_path = f'/data/data0/jiahang/nn-Meter/examples/test_transformer/implementation/nasvit/models/{name}_test'
        model.save(tf_path)
        
        tflite_fp32_path = f'/data/data0/jiahang/nn-Meter/examples/test_transformer/implementation/nasvit/models/{op_name}_jiahang.tflite'
        tf2tflite(tf_path, tflite_fp32_path)
        import shutil
        shutil.rmtree(tf_path)

        # tflite_int8_path = f'/data/data0/jiahang/nn-Meter/examples/test_transformer/implementation/nasvit/models/{name}_int8_test.tflite'
        # tf2tflite(tf_path, tflite_int8_path, quantization='int8', input_shape=[1, res, res, 3])



if __name__ == '__main__':
    export_op()
