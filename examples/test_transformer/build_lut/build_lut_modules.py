import sys
import os, json
import tensorflow as tf
import tensorflow.keras as keras
from search_space.space_utils_large import configs as search_space_config
from search_space.space_utils_large import ACT as ACT
from keras_cv_attention_models.levit.levit import mhsa_with_multi_head_position_windows, res_mlp_block, mhsa_with_multi_head_position_windows_layer_norm, res_mlp_block_layer_norm
from keras_cv_attention_models.attention_layers import conv2d_no_bias, batchnorm_with_activation, depthwise_conv2d_no_bias


ACT = 'hard-swish'


def dsconv(inputs, channel, act, strides, kernel_size, exp, use_se=False):
    inp_channel = inputs.shape[-1]
    
    if exp > 1:
        nn = conv2d_no_bias(inputs, inp_channel * exp, 1, strides=1, padding="SAME")
        nn = batchnorm_with_activation(nn, activation=act)
    else:
        nn = inputs
    
    nn = depthwise_conv2d_no_bias(inputs=nn, kernel_size=kernel_size, strides=strides, padding="SAME")
    nn = batchnorm_with_activation(nn, activation=act)

    if use_se:
        se_d = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(nn)
        se_d = conv2d_no_bias(se_d, inp_channel * exp // 4, 1, strides=1, use_bias=True)
        se_d = tf.nn.relu(se_d)
        se_d = conv2d_no_bias(se_d, inp_channel * exp, 1, strides=1, use_bias=True)
        se_d = tf.nn.relu6(se_d + 3) / 6
        nn = se_d * nn

    nn = conv2d_no_bias(nn, channel, 1, strides=1, padding="SAME")
    nn = batchnorm_with_activation(nn, activation=None)
    
    if strides == 1 and inp_channel == channel:
        nn = nn + inputs
    
    if inp_channel != channel:
        if strides == 2:
            padding = 0 if inputs.shape[-2]%2 == 0 else 1
            if padding > 0:
                inputs = keras.layers.ZeroPadding2D(padding=padding)(inputs)
        
            inputs = keras.layers.AveragePooling2D(pool_size=(2, 2), )(inputs)
        
        shortcut_conv = conv2d_no_bias(inputs, channel, kernel_size=1, strides=1, padding='SAME')
        nn = nn + shortcut_conv

    return nn


def attention_downsampling(inputs, out_channels, downsampling, dwconv = False, exp = 6, use_se = False, act = ACT):
    if len(inputs.shape) == 3:
        B, L, C = inputs.shape
        # print(inputs.shape)
        H = W = int(L**0.5)
        x = tf.reshape(inputs, [-1, H, W, C])
    else:
        x = inputs
    
    if not dwconv:
        if downsampling:
            nn = conv2d_no_bias(x, out_channels, 3, strides=2, padding='SAME', use_bias=False)
        else:
            nn = conv2d_no_bias(x, out_channels, 1, strides=1, padding='SAME', use_bias=False)
        nn = batchnorm_with_activation(nn, activation=act)
    else:
        print(x.shape)
        if downsampling:
            nn = dsconv(x, out_channels, act, 2, 3, exp, use_se=use_se)
        else:
            nn = dsconv(x, out_channels, act, 1, 3, exp, use_se=use_se)
        print(nn.shape)
        print('-'*30)
    B, H, W, C = nn.shape
    print('res trans', H)
    nn = tf.reshape(nn, [-1, int(H*W), out_channels])
    return nn


def transformer_ds(inputs, channels, ds_exp, stride, act = ACT, se = True):
    nn = attention_downsampling(inputs, channels, downsampling=stride==2, exp=ds_exp, use_se=True, dwconv=True, act=act)
    return nn


def transformer_attn(inputs, channels, v_scale, name, layer_norm = False, act = ACT):
    head_dim = 16
    num_heads = channels // head_dim # use fixed head dims here
    key_dim = head_dim
    v_dim = head_dim * v_scale

    if not layer_norm:
        nn = mhsa_with_multi_head_position_windows(inputs, channels, num_heads, key_dim, v_dim, 1, activation=act, name=name+f'layer_channel_'+str(channels))
    else:
        nn = mhsa_with_multi_head_position_windows_layer_norm(inputs, channels, num_heads, key_dim, v_dim, 1, activation=act, name=name+str(channels))
    return nn


def transformer_ffn(inputs, channels, expansion_ratio, name, layer_norm = False, act = ACT):
    if not layer_norm:
        nn = res_mlp_block(inputs, expansion_ratio, name=name+str(channels)+f'_ffn')
    else:
        nn = res_mlp_block_layer_norm(inputs, expansion_ratio, name=name+str(channels)+f'_ffn')
    return nn


def transformer_layer(inputs, channels, expansion_ratio, ds_exp, v_scale, stride, layer_index, name, layer_norm = False, act = ACT, se = True):
    if layer_index == 0:
        nn = attention_downsampling(inputs, channels, downsampling=stride==2, exp=ds_exp, use_se=True, dwconv=True)
    else:
        nn = inputs

    head_dim = 16
    num_heads = channels // head_dim # use fixed head dims here
    key_dim = head_dim
    v_dim = head_dim * v_scale

    if not layer_norm:
        nn = mhsa_with_multi_head_position_windows(nn, channels, num_heads, key_dim, v_dim, 1, activation=act, name=name+f'layer_channel_'+str(channels)+str(layer_index))
        nn = res_mlp_block(nn, expansion_ratio, name=name+str(channels)+str(layer_index)+f'_ffn')
    else:
        nn = mhsa_with_multi_head_position_windows_layer_norm(nn, channels, num_heads, key_dim, v_dim, 1, activation=act, name=name+str(channels)+str(layer_index))
        nn = res_mlp_block_layer_norm(nn, expansion_ratio, name=name+str(channels)+str(layer_index)+f'_ffn')
    return nn


def nasvit_transformer_ds(inputs, channels, ds_exp, stride, act = 'swish', se = True):
    nn = attention_downsampling(inputs, channels, downsampling=stride==2, exp=ds_exp, use_se=se, dwconv=True, act='swish')
    return nn


def nasvit_transformer_attn(inputs, channels, v_scale, name, layer_norm = False, act = 'swish'):
    head_dim = 8
    num_heads = channels // head_dim # use fixed head dims here
    key_dim = head_dim
    v_dim = head_dim * v_scale

    if not layer_norm:
        nn = mhsa_with_multi_head_position_windows(inputs, channels, num_heads, key_dim, v_dim, 1, nasvit_arch=True, activation='swish', name=name+f'layer_channel_'+str(channels))
    else:
        nn = mhsa_with_multi_head_position_windows_layer_norm(inputs, channels, num_heads, key_dim, v_dim, 1, nasvit_arch=True, activation='swish', name=name+str(channels))
    return nn


def nasvit_transformer_ffn(inputs, channels, expansion_ratio, name, layer_norm = False, act = 'swish'):
    nn = inputs
    if not layer_norm:
        for i in range(2):
            nn = res_mlp_block(nn, expansion_ratio, name=name+str(channels)+f'{i}_ffn', activation='swish')
    else:
        for i in range(2):
            nn = res_mlp_block_layer_norm(nn, expansion_ratio, name=name+str(channels)+f'{i}_ffn', activation='swish')
    return nn


def nasvit_transformer_layer(inputs, channels, expansion_ratio, ds_exp, v_scale, stride, layer_index, name, layer_norm = False, act='swish', se = True):
    
    if layer_index == 0:
        nn = nasvit_transformer_ds(inputs, channels, ds_exp, stride, se=se)
    else:
        nn = inputs
    nn = nasvit_transformer_attn(nn, channels, v_scale, name, layer_norm=layer_norm)
    nn = nasvit_transformer_ffn(nn, channels, expansion_ratio, name, layer_norm=layer_norm)
    return nn


def conv_layer(inputs, channel, expansion_ratio, kernel_size, stride, use_se, act = ACT):
    return dsconv(inputs, channel, strides=stride, kernel_size=kernel_size, exp=expansion_ratio, act=act, use_se=use_se)


def first_conv_layer(inputs, channels, stride = 2, kernel_size = 3, act = ACT):
    nn = conv2d_no_bias(inputs, channels, kernel_size, strides=stride, padding='SAME', use_bias=False)
    nn = batchnorm_with_activation(nn, activation=act)
    return nn


def mbpool_layer(inputs, channels, expansion_ratio, act = ACT):
    # efficient last stage (mbv3)
    _, N, C = inputs.shape
    out = tf.reshape(inputs, (-1, int(N**0.5), int(N**0.5), C))
    out = conv2d_no_bias(out, C * expansion_ratio, kernel_size=1, padding='SAME')
    out = batchnorm_with_activation(out, activation=act)
    out = keras.layers.GlobalAveragePooling2D(keepdims=True)(out)
    out = conv2d_no_bias(out, channels, 1, padding='SAME')

    out = tf.keras.layers.Dense(1000, dtype="float32", activation=None, name="classifier")(out)
    return out
