import os, json
import tensorflow as tf
import tensorflow.keras as keras
from conv_block import conv_layer
from transformer_block import transformer_layer, trans_ops
'''
ACT = 'hard_swish'
main_path = "/data/data0/jiahang/tflite_space/predictor_build/"
op_result = {
    "attention_downsampling": [],
    "attention": [],
    "mlp": []
}

def build_models(name, hw, cin, cout, exp, s, act, ks = None, v = None, ds = None, use_se = None):
    # return
    # if os.path.isfile(os.path.join(main_path, "kernels", f"{key}.tflite")):
    #     return        

    # print(key)
    if name == "conv": # conv
        inputs = keras.Input(shape=[hw, hw, cin], batch_size=1)
        if use_se != None:
            output = conv_layer(inputs, cout, expansion_ratio=exp, stride=s, kernel_size=ks, act=ACT, use_se=use_se)
        else:
            output = conv_layer(inputs, cout, expansion_ratio=exp, stride=s, kernel_size=ks, act=ACT)
    else: # transformer
        if ds == "ds":
            inputs = keras.Input(shape=[hw, hw, cin], batch_size=1)
            layer_index = 0
        elif ds == "nods":
            inputs = keras.Input(shape=[hw * hw, cin], batch_size=1)
            layer_index = 1
        output_op = trans_ops(inputs, cout, expansion_ratio=exp, v_scale=v, stride=s, layer_index=layer_index, name=name)
    return output_op


def add_lut_key_conv(lut_result, config, hw_lis = None, lut_result_ref = None):
    """
    Args:
        lut_result_ref (_type_, optional): only output keys not in lut_result_ref

    Returns:
        lut_result: dict of {block_config: 1}
    """
    if lut_result_ref:
        with open(os.path.join(main_path, "results", lut_result_ref), 'r') as fp:
            lut_result_ref = json.load(fp)
    else:
        lut_result_ref = None

    name = "conv"

    # layer_index = 0
    for hw in config["hw"]:
        for cin in config["cin"]:
            for cout in config["channel"]:
                for ks in config['kernel size']:
                    for exp in config['expansion_ratio']:
                        for act in [ACT]:
                            try:
                                for s in [config['stride']]:
                                    if "use_se" in config:
                                        se = config["use_se"]
                                        key = f'{name}_{hw}_{cin}_{cout}_{exp}_{s}_{act}_{ks}_{se}'
                                    else:
                                        se = None
                                        key = f'{name}_{hw}_{cin}_{cout}_{exp}_{s}_{act}_{ks}'
                                    
                                    if key not in lut_result:
                                        if lut_result_ref and key in lut_result_ref:
                                            print(1)
                                            continue
                                        else:
                                            build_models(key, name, hw, cin, cout, exp, s, act, ks=ks, use_se=se)
                                            lut_result[key] = [[hw, hw, cin]]
                                            # return lut_result
                            except:
                                import pdb; pdb.set_trace()
    # layer_index > 0
    if config['name'] == 'first_conv':
        return lut_result
    for hw in config["hw_out"]:
        for cin in config["channel"]:
            cout = cin
            for ks in config['kernel size']:
                for exp in config['expansion_ratio']:
                    for act in [ACT]:
                        if "use_se" in config:
                            se = config["use_se"]
                            key = f'{name}_{hw}_{cin}_{cout}_{exp}_1_{act}_{ks}_{se}'
                        else:
                            se = None
                            key = f'{name}_{hw}_{cin}_{cout}_{exp}_1_{act}_{ks}'
                        if key not in lut_result:
                            if lut_result_ref and key in lut_result_ref:
                                print(1)
                                continue
                            else:
                                build_models(key, name, hw, cin, cout, exp, s, act, ks=ks, use_se=se)
                                lut_result[key] = [[hw, hw, cin]]
                                # return lut_result
    return lut_result


def add_lut_key_transformer(lut_result, config, hw_lis = None, lut_result_ref = None):
    """
    Args:
        lut_result_ref (_type_, optional): only output keys not in lut_result_ref

    Returns:
        lut_result: dict of {block_config: 1}
    """
    name = "transformer"

    # layer_index = 0 / ds
    ds = "ds"
    for hw in config["hw"]:
        for cin in config["cin"]:
            for cout in config["channel"]:
                for exp in config['expansion_ratio']:
                    for act in [ACT]:
                        for s in [config['stride']]:
                            for v in config['v_scale']:
                                new_key = f'{name}_{hw}_{cin}_{cout}_{exp}_{s}_{act}_{v}_{ds}'
                                if new_key not in lut_result:
                                    logs = build_models(name, hw, cin, cout, exp, s, act, v=v, ds=ds)
                                    lut_result[new_key] = 1
                                    # import pdb; pdb.set_trace()
                                    for k in logs:
                                        if logs[k] not in op_result[k]:
                                            op_result[k].append(logs[k])

    ds = "nods"
    for hw in config["hw_out"]:
        for cout in config["channel"]:
            cin = cout
            for exp in config['expansion_ratio']:
                for act in [ACT]:
                    for v in config['v_scale']:
                        new_key = f'{name}_{hw}_{cin}_{cout}_{exp}_1_{act}_{v}_{ds}' #TODO
                        if new_key not in lut_result:
                            if cin != cout:
                                continue
                            else:
                                logs = build_models(name, hw, cin, cout, exp, 1, act, v=v, ds=ds)
                                lut_result[new_key] = 1
                                for k in logs:
                                    if logs[k] not in op_result[k]:
                                        op_result[k].append(logs[k])
    return lut_result


lut_result = {}


from space_utils import configs as c3t3_config
from space_utils_2c4t import configs as c2t4_config
from space_utils_4c2t import configs as c4t2_config
from space_utils_attnnas import configs as attnnas_config
from space_utils_spacev1 import configs as spacev1_config
for configs in [c3t3_config, c2t4_config, c4t2_config, attnnas_config, spacev1_config]:
    for i in range(1, 7):
        if configs[i]["block_type"] == 0:
            pass
            # lut_result = add_lut_key_conv(lut_result, configs[i], "op_all.json")
        else:
            lut_result = add_lut_key_transformer(lut_result, configs[i], "op_all.json")
print(len(lut_result))

# import pdb; pdb.set_trace()
# res = {}
# for key, value in lut_result.items():
#     res[key] = {
#         "converted_model": f"/data1/jiahang/working/pixel6_supernet_workspace/predictor_build/models/{key}.tflite",
#         'shapes': value
#     }
#     # res[key] = 1

with open("/data/data0/jiahang/vit_lut/block_lut.json", 'w') as fp:
    json.dump(op_result, fp, indent=4)
    # json.dump(res, fp, indent=4)

# nohup /home/v-chentang/anaconda3/bin/python vit_lut/build_lut.py > trans_lut_log.txt 2>&1 &
# [1] 10136

# scp /data/data0/jiahang/tflite_space/predictor_build/kernels/* jiahang@10.172.141.20:/data1/jiahang/working/pixel6_supernet_workspace/predictor_build/models/
# scp /data/data0/jiahang/tflite_space/predictor_build/results_pixel6/lut_v6.json jiahang@10.172.141.20:/data1/jiahang/working/pixel6_supernet_workspace/predictor_build/results/

'''
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