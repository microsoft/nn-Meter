
import pstats
import tensorflow as tf
import random
import shutil
import os
import math
import pickle
import json
import numpy as np
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
import tensorflow.keras as keras
from tensorflow.keras.layers import BatchNormalization
from keras_cv_attention_models.common_layers import hard_swish
import multiprocessing
# from tflite import ADB, PIXEL4_SERIAL_NUMBER

from keras_cv_attention_models.levit.levit import mhsa_with_multi_head_position_windows, res_mlp_block, mhsa_with_multi_head_position_windows_layer_norm, res_mlp_block_layer_norm
from nasvit_tf import dsconv, attention_downsampling, first_conv


ACT = 'swish'
# ACT = 'hard_swish'
LAYER_NORM = True
DOWNSAMPLING = [True, True, True, False, True, True]
TALKING_HEAD = True
main_path = "/data/data0/jiahang/tflite_space/predictor_build/"

def conv_layer(inputs, channel, expansion_ratio, kernel_size, stride, use_se, act=ACT):
    return dsconv(inputs, channel, strides=stride, kernel_size=kernel_size, exp=expansion_ratio, act=act, use_se=use_se)


def transformer_layer(inputs, channels, expansion_ratio, ds, v_scale, stride, layer_index, se, layer_norm = True, reshape=False, name = ""):
    nn = inputs
    if reshape:
        _, H, W, C = nn.shape
        nn = tf.reshape(inputs, (-1, H*W, C))
    
    if layer_index == 0:
        se = True
        
        nn = attention_downsampling(nn, channels, ds, dw_downsampling=True, exp=6, use_se=se)

    if not layer_norm:
        nn = mhsa_with_multi_head_position_windows(nn, channels, channels//8, 8, 8 * v_scale, 1, nasvit_arch=True, activation=ACT, name=f'stage_layer_channel')
        for _ in range(2):
            nn = res_mlp_block(nn, expansion_ratio, activation=ACT, name='ffn')
    else:
        nn = mhsa_with_multi_head_position_windows_layer_norm(nn, channels, channels//8, 8, 8 * v_scale, 1, nasvit_arch=True, activation=ACT, name='stage_layer_channel')
        for _ in range(2):
            nn = res_mlp_block_layer_norm(nn, expansion_ratio, activation=ACT, name='ffn')
    return nn


def build_models(key, name, hw, cin, cout, exp, s, act, ks = None, v = None, ds = None, use_se = None, reshape = False):
    # return
    if os.path.isfile(os.path.join(main_path, "nasvit_lut", f"{key}.tflite")): return
    
    if name == "MBlock": # mobile conv 
        inputs = keras.Input(shape=[hw, hw, cin], batch_size=1)
        output = conv_layer(inputs, cout, expansion_ratio=exp, stride=s, kernel_size=ks, act=act, use_se=use_se)
        
    else: # transformer
        if reshape: # the first transformer layer in the first transformer block
            inputs = keras.Input(shape=[hw, hw, cin], batch_size=1)
            layer_index = 0
        else:
            inputs = keras.Input(shape=[hw * hw, cin], batch_size=1)
            layer_index = 1
        output = transformer_layer(inputs, cout, expansion_ratio=exp, ds=ds, v_scale=v, stride=s, layer_index=layer_index, se=True, name=name, layer_norm=layer_norm, reshape=reshape)

    model = keras.Model(inputs=inputs, outputs=output)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()
    converted_model = os.path.join(main_path, "nasvit_lut", f"{key}.tflite")
    open(converted_model, 'wb').write(tflite_model)


# NasVit
nasvit_config = [
    # name       depths              width                   expansion_ratio ks          stride  se          hw candidate
    ("Conv",     [1],                [16, 24],               [1],            [3],        2,      False,      [192, 224, 256, 288]),
    ("MBlock",   [1, 2],             [16, 24],               [1],            [3, 5],     1,      False,      np.unique([item // 2 for item in [192, 224, 256, 288]])), # 112
    ("MBlock",   [3, 4, 5],          [24, 32],               [4, 5, 6],      [3, 5],     2,      False,      np.unique([item // 2 for item in [192, 224, 256, 288]])), # 112
    ("MBlock",   [3, 4, 5, 6],       [32, 40],               [4, 5, 6],      [3, 5],     2,      True,       np.unique([item // 4 for item in [192, 224, 256, 288]])), # 56
    ("Trans",    [3, 4, 5, 6],       [64, 72],               [1, 2],         [None],     2,      False,      np.unique([item // 8 for item in [192, 224, 256, 288]])), # 28
    ("Trans",    [3, 4, 5, 6, 7, 8], [112, 120, 128],        [1, 2],         [None],     2,      False,      np.unique([item // 16 for item in [192, 224, 256, 288]])), # 14
    ("Trans",    [3, 4, 5, 6, 7, 8], [160, 168, 176, 184],   [1, 2],         [None],     1,      False,      np.unique([item // 32 for item in [192, 224, 256, 288]])), # 7
    ("Trans",    [3, 4, 5, 6],       [208, 216, 224],        [1, 2],         [None],     2,      False,      np.unique([item // 32 for item in [192, 224, 256, 288]])), # 7
    ("Conv",     [1],                [1792, 1984],           [6],            [None],     1,      False,      np.unique([item // 64 for item in [192, 224, 256, 288]])) # 3
]

lut_result = {}
for supernet_config in [nasvit_config]:
    for i, item in enumerate(supernet_config):
        name, depths, widths, exp_ratios, kses, stride, se, hws = item
        cins = supernet_config[i - 1][2] if i > 0 else [3]
        hw_outs = supernet_config[i + 1][-1] if i < len(supernet_config) - 1 else [None]
        if name not in ['MBlock', 'Trans']: continue
        # import pdb; pdb.set_trace()
        
        # MBlock
        # layer_index = 0
        for hw in hws:
            for cin in cins:
                for cout in widths:
                    for ks in kses:
                        for exp in exp_ratios:
                            for act in [ACT]:
                                if name == "MBlock":
                                    key = f'{name}_{hw}_{cin}_{cout}_{exp}_{stride}_{act}_{ks}_{"se" if se else "nose"}'
                                    if key not in lut_result:
                                        build_models(key, name, hw, cin, cout, exp, stride, act, ks=ks, use_se=se)
                                        lut_result[key] = [[hw, hw, cin]]
                                else:
                                    # for nasvit, v_scale = 4, ds_exp = 6, norm = ln
                                    key = f'{name}_{hw}_{cin}_{cout}_{exp}_{stride}_{act}_{4}_ds_{6}_ln'
                                    # key = f'{name}_{hw}_{cin}_{cout}_{exp}_1_{act}_{v}_{ds}_ln'
                                    
        # layer_index > 0
        for hw in hw_outs:
            for cin in widths:
                cout = cin
                for ks in kses:
                    for exp in exp_ratios:
                        for act in [ACT]:
                            
                            if name == "MBlock":
                                key = f'{name}_{hw}_{cin}_{cout}_{exp}_1_{act}_{ks}_{"se" if se else "nose"}'
                                if key not in lut_result:
                                    build_models(key, name, hw, cin, cout, exp, 1, act, ks=ks, use_se=se)
                                    lut_result[key] = [[hw, hw, cin]]
                                    # return lut_result
                            else:
                                key = f'{name}_{hw}_{cin}_{cout}_{exp}_1_{act}_{4}_nods_ln'
                                if key not in lut_result:
                                    build_models(key, name, hw, cin, cout, exp, 1, act, ks=ks, use_se=se)
                                    lut_result[key] = [[hw, hw, cin]]

print(len(lut_result))
print(list(lut_result.keys()))

res = {}
for key, value in lut_result.items():
    # res[key] = {
    #     "converted_model": f"/data1/jiahang/working/pixel6_supernet_workspace/predictor_build/models/{key}.tflite",
    #     'shapes': value
    # }
    res[key] = 1

with open(os.path.join(main_path, "results_pixel6", f"nasvit_lut_v1.json"), 'w') as fp:
    # json.dump({"lut": res}, fp, indent=4)
    json.dump(res, fp, indent=4)


# nohup python /data/data0/jiahang/nn-Meter/examples/test_transformer/implementation/nasvit/build_lut.py > nasvit_lut_log.txt 2>&1 &
# [1] 30909