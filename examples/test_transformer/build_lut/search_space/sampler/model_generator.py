
import tensorflow as tf
import random
import shutil
import os
import math
import pickle
import sys
from sampler import arch_sampling_unpacked, STAGE, DOWNSAMPLING
sys.path.append("/data1/jiahang/working/pixel6_fp32_workspace/nn-Meter/examples/test_transformer/cv_attention")

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


from keras_cv_attention_models.common_layers import hard_swish

from keras_cv_attention_models.levit.levit import mhsa_with_multi_head_position_windows, res_mlp_block, mhsa_with_multi_head_position_windows_layer_norm, res_mlp_block_layer_norm

ACT = 'hard_swish'
LAYER_NORM = False


def dsconv(inputs, channel, act, strides, kernel_size, exp):
    inp_channel = inputs.shape[-1]
    
    nn = conv2d_no_bias(inputs, inp_channel * exp, 1, strides=1, padding="same")
    nn = batchnorm_with_activation(nn, activation=act)
    nn = depthwise_conv2d_no_bias(inputs=nn, kernel_size=kernel_size, strides=strides, padding='same')

    nn = batchnorm_with_activation(nn, activation=act)
    nn = conv2d_no_bias(nn, channel, 1, strides=1, padding="same")
    return nn


def first_conv(inputs, channels, act):
    nn = conv2d_no_bias(inputs, channels, 3, strides=2, padding='same', use_bias=False)
    nn = batchnorm_with_activation(nn, activation=act)
    nn = depthwise_conv2d_no_bias(nn, kernel_size=3, strides=1, padding='same',use_bias=False)
    nn = batchnorm_with_activation(nn, activation=act)
    nn = conv2d_no_bias(nn, channels, 1, strides=1, padding='same', use_bias=False)
    return nn


def attention_downsampling(inputs, out_channels, downsampling):
    B, L, C = inputs.shape
    # print(inputs.shape)
    H = W = int(L**0.5)
    x = tf.reshape(inputs, [-1, H, W, C])
    if downsampling:
        nn = conv2d_no_bias(x, out_channels, 3, strides=2, padding='same', use_bias=False)
    else:
        nn = conv2d_no_bias(x, out_channels, 1, strides=1, padding='same', use_bias=False)
    nn = batchnorm_with_activation(nn, activation=ACT)
    B, H, W, C = nn.shape
    nn = tf.reshape(nn, [-1, int(H*W), out_channels])
    
    # x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
    # x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
    # x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
    # x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
    # if x0.shape != x1.shape and H == 7:

    #     nn = tf.reshape(x0, [-1, int(16), C])
        
    #     nn = tf.keras.layers.Dense(out_channels)(nn)
    #     return nn
    # x = tf.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
    # nn = tf.reshape(x, [-1, int(H*W/4), 4 * C])  # B H/2*W/2 4*C
    # if LAYER_NORM:
    #     nn = layer_norm(nn)
    # nn = tf.keras.layers.Dense(out_channels)(nn)
    # if not LAYER_NORM:
    #     nn = BatchNormalization()(nn)
    # print(nn.shape)
    return nn


def build_model_new(res, inputs, channels, depths, conv_ratio, kr_size, mlp_ratio, num_heads, window_size, qk_scale, v_scale, downsampling, stages):
    nn = first_conv(inputs, 128//8, ACT)
    res = res//2
    c = 128//8

    num_conv_stage = stages.count('C') # 3
    num_transformer_stage = stages.count('T') # 3
    
    for stage in range(len(stages)):
        for layer in range(depths[stage]):
            if stage < num_conv_stage:
                conv_offset = sum(depths[:stage]) + layer
                nn = dsconv(nn, channels[stage], ACT, 2 if layer == 0 and downsampling[stage] else 1, kr_size[conv_offset], conv_ratio[conv_offset])
            
            if stage >= num_conv_stage:
                if stage == num_conv_stage and layer == 0:
                    _, H, W, C = nn.shape
                    nn = tf.reshape(nn, (-1, H*W, C))
                
                if layer == 0:
                    nn = attention_downsampling(nn, channels[stage], downsampling[stage])
                
                tranformer_offset = sum(depths[num_conv_stage:stage]) + layer
                if not LAYER_NORM:
                    nn = mhsa_with_multi_head_position_windows(nn, channels[stage], num_heads[tranformer_offset], int(channels[stage]//num_heads[tranformer_offset]//qk_scale[tranformer_offset]), int(channels[stage]//num_heads[tranformer_offset]*v_scale[tranformer_offset]), window_size[tranformer_offset], name=f'stage_{stage}_layer_channel_'+str(channels[stage])+str(layer))
                    nn = res_mlp_block(nn, mlp_ratio[tranformer_offset], name=str(channels[stage])+str(layer)+'_ffn')
                else:
                    nn = mhsa_with_multi_head_position_windows_layer_norm(nn, channels[stage], num_heads[tranformer_offset], int(channels[stage]//num_heads[tranformer_offset]//qk_scale[tranformer_offset]), int(channels[stage]//num_heads[tranformer_offset]*v_scale[tranformer_offset]), window_size[tranformer_offset], name=str(channels[stage])+str(layer))
                    nn = res_mlp_block_layer_norm(nn, mlp_ratio[tranformer_offset], name=str(channels[stage])+str(layer)+'_ffn')
    
    out = tf.keras.layers.Dense(960, dtype="float32", activation=None, name="pre_head")(nn)
    out = batchnorm_with_activation(out, activation=ACT, name="pre_head_bn")
    out = tf.keras.layers.GlobalAveragePooling1D()(out)
    out = tf.keras.layers.Dense(1280, dtype="float32", activation=hard_swish, name="head0")(out)
    out = tf.keras.layers.Dense(1000, dtype="float32", activation=None, name="head")(out)
    model = tf.keras.models.Model(inputs, out)
    return model
                
if __name__ == '__main__':
    resolution, channels, depths, conv_ratio, kr_size, mlp_ratio, num_heads, window_size, qk_scale, v_scale = arch_sampling_unpacked(mode='uniform')

    input_shape = (resolution, resolution, 3)
    inputs = tf.keras.layers.Input(input_shape)
    model = build_model_new(resolution, inputs, channels, depths, conv_ratio, kr_size, mlp_ratio, num_heads, window_size, qk_scale, v_scale, DOWNSAMPLING, STAGE)
    model(tf.random.normal([1, resolution, resolution, 3]))
    

    from nn_meter.builder.backends import connect_backend
    from nn_meter.predictor import load_latency_predictor
    from nn_meter.builder import builder_config
    from nn_meter.builder.nn_generator.tf_networks.utils import get_inputs_by_shapes
    output_path = "/data/jiahang/working/nn-Meter/examples/test_transformer"
    output_name = os.path.join(output_path, "test")

    workspace = "/data1/jiahang/working/pixel6_fp32_workspace/"
    builder_config.init(workspace)
    backend = connect_backend(backend_name='tflite_cpu')
    model(get_inputs_by_shapes([[*input_shape]]))
    tf.keras.models.save_model(model, output_name)

    res = backend.profile_model_file(output_name, output_path, input_shape=[[*input_shape]])
    print(res)
