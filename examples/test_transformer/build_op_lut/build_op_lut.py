import os, json
import tensorflow as tf
import tensorflow.keras as keras
from conv_block import conv_layer
from transformer_block import transformer_layer, trans_ops


import json
with open("/data/data0/jiahang/vit_lut/block_lut.json", "r") as fp:
    sone = json.load(fp)


op_res = {
    'conv': [],
    'bn': [],
    'hswish': [],
    'reshape_4Dto3D': [],
    'reshape_3Dto3D': [],
    'fc_3D': [],
    'bn_3D': [],
    'reshape_3Dto4D': [],
    'transpose_0213': [],
    'split_3out': [],
    'reshape_4Dto4D': [],
    'dwconv_4D': [],
    'matmul_4D_transpose': [],
    'element_divide': [],
    'embedding': [],
    'transpose_0231': [],
    'fc_4D': [],
    'transpose_0312': [],
    'softmax_4D': [],
    'matmul_4D_notrans': [],
    'hswish_3D': [],
    'ln_3D': [],
    'add_3D': []
}

for attn_ds_config in sone["attention_downsampling"]:
    input_shape, output_shape, cout, ds, dwconv = attn_ds_config
    b, h, w, cin = input_shape
    assert b == 1
    assert h == w
    hw = h
    if ds:
        op_res['conv'].append([hw, cin, cout, 3, 2])
        hw = hw // 2 if hw % 2 == 0 else hw // 2 + 1
        op_res['bn'].append([hw, cout])
        op_res['hswish'].append([hw, cout])
        op_res['reshape_4Dto3D'].append([b, hw, hw, cout, b, hw * hw, cout]) # [B, HW, HW, COUT] -> [B, HW * HW, COUT]
    else:
        op_res['conv'].append([hw, cin, cout, 1, 1])
        op_res['bn'].append([hw, cout])
        op_res['hswish'].append([hw, cout])
        op_res['reshape_4Dto3D'].append([b, hw, hw, cout, b, hw * hw, cout]) # [B, HW, HW, COUT] -> [B, HW * HW, COUT]
    assert output_shape == [1, hw * hw, cout]

talking_head = True
activation = 'hard_swish'
layer_norm = False
for attn_config in sone["attention"]:
    # windowsize = 1
    input_shape, output_shape, cout, num_heads, key_dim, v_dim, window_size, _ = attn_config
    b, hw, cin = input_shape
    embed_dim = int(key_dim * num_heads)
    embed_dim_v = int(v_dim * num_heads)
    assert b == 1
    assert cin == cout
    

    if window_size > 1:
        ww = window_size * window_size
        _b = 4 # resume the window_size = hw // 2
        op_res['reshape_3Dto3D'].append([b, hw, cin, _b, ww, cin]) # [B, HW, CIN] -> [_B, WW, CIN]        
    else:
        ww = hw
        _b = b

    qkv_dim = int(2 * embed_dim + embed_dim_v)
    op_res['fc_3D'].append([_b, ww, cin, qkv_dim]) # [_B, WW, CIN] -> [_B, WW, QKV_DIM]
    op_res['bn_3D'].append([_b, ww, qkv_dim]) # [_B, WW, QKV_DIM] -> [_B, WW, QKV_DIM]
    op_res['reshape_3Dto4D'].append([_b, ww, qkv_dim, num_heads]) # [_B, WW, QKV_DIM] -> [_B, WW, NUM_HEADS, QKV_DIM//NUM_HEADS]
    op_res['transpose_0213'].append([_b, ww, num_heads, qkv_dim // num_heads]) # [_B, WW, NUM_HEADS, QKV_DIM//NUM_HEADS] -> [_B, NUM_HEADS, WW, QKV_DIM//NUM_HEADS]
    op_res['split_3out'].append([_b, num_heads, ww, qkv_dim // num_heads, key_dim, key_dim, v_dim]) # [_B, NUM_HEADS, WW, QKV_DIM//NUM_HEADS] -> [_B, NUM_HEADS, WW, K_DIM], [..., K_DIM], [..., V_DIM]
    op_res['reshape_4Dto4D'].append([_b, num_heads, ww, v_dim, hw ** 0.5, hw ** 0.5, embed_dim_v]) # [_B, NUM_HEADS, WW, V_DIM] -> [_B, H, W, NUM_HEADS * V_DIM]
    op_res['dwconv_4D'].append([_b, hw ** 0.5, embed_dim_v, embed_dim_v, 3, 1]) # [_B, H, W, NUM_HEADS * V_DIM] -> [_B, H, W, NUM_HEADS * V_DIM]
    op_res['reshape_4Dto4D'].append([_b, hw ** 0.5, hw ** 0.5, embed_dim_v, num_heads, ww, v_dim]) # [_B, H, W, NUM_HEADS * V_DIM] -> [_B, NUM_HEADS, WW, V_DIM]

    # scaled_dot_product_attention(qq, kk, vv, key_dim, 1, output_dim=output_dim, activation=activation, name=name+'attn', talking_head=True)
    op_res['matmul_4D_transpose'].append([_b, num_heads, ww, key_dim, key_dim, True]) # [[_B, NUM_HEADS, WW, K_DIM], [_B, NUM_HEADS, WW, K_DIM]] -> [_B, NUM_HEADS, WW, WW]
    op_res['element_divide'].append([_b, num_heads, ww, ww, key_dim ** 0.5]) # [_B, NUM_HEADS, WW, WW] -> [_B, NUM_HEADS, WW, WW]
    op_res['embedding'].append([num_heads, hw, key_dim, True]) # [_B, NUM_HEADS, WW, WW] -> [_B, NUM_HEADS, WW, WW]

    if talking_head:
        op_res['transpose_0231'].append([_b, num_heads, hw, key_dim]) # [_B, NUM_HEADS, WW, WW] -> [_B, WW, WW, NUM_HEADS]
        op_res['fc_4D'].append([_b, ww, ww, num_heads, num_heads]) # [_B, WW, WW, NUM_HEADS] -> [_B, WW, WW, NUM_HEADS]
        op_res['transpose_0312'].append([_b, num_heads, hw, key_dim]) # [_B, WW, WW, NUM_HEADS] -> [_B, NUM_HEADS, WW, WW]
    op_res['softmax_4D'].append([_b, ww, ww, num_heads, num_heads]) # [_B, NUM_HEADS, WW, WW] -> [_B, NUM_HEADS, WW, WW]
    if talking_head:
        op_res['transpose_0231'].append([_b, num_heads, hw, key_dim]) # [_B, NUM_HEADS, WW, WW] -> [_B, WW, WW, NUM_HEADS]
        op_res['fc_4D'].append([_b, ww, ww, num_heads, num_heads]) # [_B, WW, WW, NUM_HEADS] -> [_B, WW, WW, NUM_HEADS]
        op_res['transpose_0312'].append([_b, num_heads, hw, key_dim]) # [_B, WW, WW, NUM_HEADS] -> [_B, NUM_HEADS, WW, WW]
    
    op_res['matmul_4D_notrans'].append([_b, num_heads, ww, ww, v_dim, False]) # [[_B, NUM_HEADS, WW, WW], [_B, NUM_HEADS, WW, V_DIM]] -> [_B, NUM_HEADS, WW, V_DIM]
    op_res['transpose_0213'].append([_b, num_heads, hw, key_dim]) # [_B, NUM_HEADS, WW, V_DIM] -> [_B, WW, NUM_HEADS, V_DIM]
    op_res['reshape_4Dto3D'].append([_b, ww, num_heads, v_dim, _b, ww, num_heads * v_dim]) # [_B, WW, NUM_HEADS, V_DIM] -> [_B, WW, NUM_HEADS * V_DIM]
    if activation:
        op_res['hswish_3D'].append([_b, ww, num_heads * v_dim]) # [_B, WW, NUM_HEADS * V_DIM] -> [_B, WW, NUM_HEADS * V_DIM]
    op_res['fc_3D'].append([_b, ww, num_heads * v_dim, cout]) # [_B, WW, NUM_HEADS * V_DIM] -> [_B, WW, COUT]
    if not layer_norm:
        op_res['ln_3D'].append([_b, ww, cout]) # [_B, WW, COUT] -> [_B, WW, COUT]
    op_res['reshape_3Dto3D'].append([_b, ww, cout, b, hw, cout]) # [_B, WW, COUT] -> [B, HW, COUT]


for mlp_config in sone["mlp"]:
    input_shape, output_shape, mlp_ratio, ACT = mlp_config
    b, hw, cin = input_shape # [B, HW, CIN]
    op_res['fc_3D'].append([b, hw, cin, cin * mlp_ratio]) # [B, HW, CIN] -> [B, HW, CIN * MLP_RATIO]
    op_res['bn_3D'].append([b, hw, cin * mlp_ratio]) # [B, HW, CIN * MLP_RATIO] -> [B, HW, CIN * MLP_RATIO]
    op_res['hswish_3D'].append([b, hw, cin * mlp_ratio]) # [B, HW, CIN * MLP_RATIO] -> [B, HW, CIN * MLP_RATIO]
    op_res['fc_3D'].append([b, hw, cin * mlp_ratio, cin]) # [B, HW, CIN * MLP_RATIO] -> [B, HW, CIN]
    op_res['bn_3D'].append([b, hw, cin]) # [B, HW, CIN] -> [B, HW, CIN]
    op_res['add_3D'].append([b, hw, cin]) # [B, HW, CIN] -> [B, HW, CIN]

import numpy as np
# import pdb; pdb.set_trace()
for key, value in op_res.items():
    value_new = []
    for item in value:
        item_new = [int(i) for i in item]
        if item_new not in value_new:
            value_new.append(item_new)
    # import pdb; pdb.set_trace()
    print(key, len(value), len(value_new))
    op_res[key] = value_new
    



with open("/data/data0/jiahang/vit_lut/block_op_lut.json", 'w') as fp:
    json.dump(op_res, fp, indent=4)