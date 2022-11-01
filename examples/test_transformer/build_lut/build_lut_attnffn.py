import sys
import os, json
import tensorflow as tf
import tensorflow.keras as keras
from search_space.space_utils_large import configs as search_space_config
from search_space.space_utils_large import ACT as ACT
from keras_cv_attention_models.levit.levit import mhsa_with_multi_head_position_windows, res_mlp_block, mhsa_with_multi_head_position_windows_layer_norm, res_mlp_block_layer_norm
from build_lut import conv_layer, first_conv_layer, mbpool_layer, add_lut_key_conv, add_lut_key_mbpool, add_lut_key_firstconv
sys.path.append("/home/v-chentang/vit/")
from fusednas_sampling2 import attention_downsampling, dsconv


mark = sys.argv[1]
layer_norm = True if mark == 'ln' or mark == 'nasvit' else False
nasvit_arch = True if mark == 'nasvit' else False
ACT = 'swish' if nasvit_arch else ACT
main_path = "/data/data0/jiahang/tflite_space/predictor_build/"
# print(mark, layer_norm)


def transformer_ds(inputs, channels, ds_exp, stride):
    nn = attention_downsampling(inputs, channels, downsampling=stride==2, exp=ds_exp, use_se=True, dwconv=True)
    return nn


def transformer_attn(inputs, channels, v_scale, name, layer_norm = False):
    head_dim = 16
    num_heads = channels // head_dim # use fixed head dims here
    key_dim = head_dim
    v_dim = head_dim * v_scale

    if not layer_norm:
        nn = mhsa_with_multi_head_position_windows(inputs, channels, num_heads, key_dim, v_dim, 1, activation=ACT, name=name+f'layer_channel_'+str(channels))
    else:
        nn = mhsa_with_multi_head_position_windows_layer_norm(inputs, channels, num_heads, key_dim, v_dim, 1, activation=ACT, name=name+str(channels))
    return nn


def transformer_ffn(inputs, channels, expansion_ratio, name, layer_norm = False):
    if not layer_norm:
        nn = res_mlp_block(inputs, expansion_ratio, name=name+str(channels)+f'_ffn')
    else:
        nn = res_mlp_block_layer_norm(inputs, expansion_ratio, name=name+str(channels)+f'_ffn')
    return nn


def nasvit_transformer_ds(inputs, channels, ds_exp, stride, se = True):
    from nasvit_tf import attention_downsampling as nasvit_attn_ds
    nn = nasvit_attn_ds(inputs, channels, downsampling=stride==2, exp=ds_exp, use_se=se, dwconv=True, act='swish')
    return nn


def nasvit_transformer_attn(inputs, channels, v_scale, name, layer_norm = False):
    head_dim = 8
    num_heads = channels // head_dim # use fixed head dims here
    key_dim = head_dim
    v_dim = head_dim * v_scale

    if not layer_norm:
        nn = mhsa_with_multi_head_position_windows(inputs, channels, num_heads, key_dim, v_dim, 1, nasvit_arch=True, activation='swish', name=name+f'layer_channel_'+str(channels))
    else:
        nn = mhsa_with_multi_head_position_windows_layer_norm(inputs, channels, num_heads, key_dim, v_dim, 1, nasvit_arch=True, activation='swish', name=name+str(channels))
    return nn


def nasvit_transformer_ffn(inputs, channels, expansion_ratio, name, layer_norm = False):
    if not layer_norm:
        for i in range(2):
            nn = res_mlp_block(inputs, expansion_ratio, name=name+str(channels)+f'{i}_ffn')
    else:
        for i in range(2):
            nn = res_mlp_block_layer_norm(inputs, expansion_ratio, name=name+str(channels)+f'{i}_ffn')
    return nn


def build_models(key, name, hw, cin, cout, exp, s, act, ks = None, v = None, ds = None, use_se = None, ds_exp = None):
    return
    print(key)
    if os.path.isfile(os.path.join(main_path, "common_layer", f"{key}.tflite")):
        return
    if os.path.isfile(os.path.join(main_path, "trans_blocks", f"{key}.tflite")):
        return 
    if os.path.isfile(os.path.join(main_path, "nasvit_lut", f"{key}.tflite")):
        return 

    if name == "firstconv":
        inputs = keras.Input(shape=[hw, hw, cin], batch_size=1)
        output = first_conv_layer(inputs, cout, stride=s, kernel_size=ks, act=act)

    if name == "conv": # conv
        if not layer_norm: return
        inputs = keras.Input(shape=[hw, hw, cin], batch_size=1)
        use_se = use_se or False
        output = conv_layer(inputs, cout, expansion_ratio=exp, stride=s, kernel_size=ks, act=act, use_se=use_se)

    elif name == "transds":
        se = use_se if nasvit_arch else True
        cls = nasvit_transformer_ds if nasvit_arch else transformer_ds
        inputs = keras.Input(shape=[hw * hw, cin], batch_size=1)
        output = cls(inputs, cout, ds_exp=ds_exp, stride=s, se=se)
    
    elif name == "transattn": # transformer attention layer
        cls = nasvit_transformer_attn if nasvit_arch else transformer_attn
        inputs = keras.Input(shape=[hw * hw, cin], batch_size=1)
        output = cls(inputs, cout, v_scale=v, name=name, layer_norm=layer_norm)
    
    elif name == "transffn": # transformer attention layer
        cls = nasvit_transformer_ffn if nasvit_arch else transformer_ffn
        inputs = keras.Input(shape=[hw * hw, cin], batch_size=1)
        output = cls(inputs, cout, expansion_ratio=exp, name=name, layer_norm=layer_norm)

    elif name == "mbpool":
        inputs = keras.Input(shape=[hw * hw, cin], batch_size=1)
        output = mbpool_layer(inputs, cout, expansion_ratio=exp, act=act)

    model = keras.Model(inputs=inputs, outputs=output)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()
    dir_name = "nasvit_lut" if nasvit_arch else "trans_blocks"
    converted_model = os.path.join(main_path, dir_name, f"{key}.tflite")
    open(converted_model, 'wb').write(tflite_model)


def add_lut_key_transformer(lut_result, config, hw_lis = None, lut_result_ref = None, stage_ds_se = True):
    """
    Args:
        lut_result_ref (_type_, optional): only output keys not in lut_result_ref

    Returns:
        lut_result: dict of {block_config: 1}
    """
    if lut_result_ref:
        with open(lut_result_ref, 'r') as fp:
            lut_result_ref = json.load(fp)
    else:
        lut_result_ref = None

    # layer_index = 0 / ds
    ds = "ds"
    for hw in config["hw"]:
        for cin in config["cin"]:
            for cout in config["channel"]:
                for exp in config['expansion_ratio']:
                    for act in [ACT]:
                        for s in [config['stride']]:
                            for v in config['v_scale']:
                                for ds_exp in config['downsample_expansion_ratio']:
                                    # transds
                                    nasvit_mark = f"nasvit_{'se' if stage_ds_se else 'nose'}_" if nasvit_arch else ""
                                    new_key = f'{nasvit_mark}transds_{hw}_{cin}_{cout}_{s}_{ds_exp}'
                                    if new_key not in lut_result:
                                        if lut_result_ref and new_key in lut_result_ref:
                                            print(new_key)
                                            continue
                                        else:
                                            build_models(new_key, "transds", hw, cin, cout, exp, s, act, use_se=stage_ds_se, v=v, ds=ds, ds_exp=ds_exp)
                                            lut_result[new_key] = [[hw, hw, cin]]

                                    # transattn
                                    nasvit_mark = f"nasvit_" if nasvit_arch else ""
                                    new_key = f'{nasvit_mark}transattn_{hw}_{cout}_{act}_{v}_{"ln" if layer_norm else "bn"}'
                                    if new_key not in lut_result:
                                        if lut_result_ref and new_key in lut_result_ref:
                                            print(new_key)
                                            continue
                                        else:
                                            build_models(new_key, "transattn", hw, cout, cout, exp, s, act, v=v, ds=ds, ds_exp=ds_exp)
                                            lut_result[new_key] = [[hw, hw, cin]]
                                    
                                    # transffn
                                    nasvit_mark = f"nasvit_" if nasvit_arch else ""
                                    new_key = f'{nasvit_mark}transffn_{hw}_{cout}_{exp}_{act}_{"ln" if layer_norm else "bn"}'
                                    if new_key not in lut_result:
                                        if lut_result_ref and new_key in lut_result_ref:
                                            print(new_key)
                                            continue
                                        else:
                                            build_models(new_key, "transffn", hw, cout, cout, exp, s, act, v=v, ds=ds, ds_exp=ds_exp)
                                            lut_result[new_key] = [[hw, hw, cin]]


    ds = "nods"
    for hw in config["hw_out"]:
        for cout in config["channel"]:
            cin = cout
            for exp in config['expansion_ratio']:
                for act in [ACT]:
                    for v in config['v_scale']:
                        # transattn
                        nasvit_mark = f"nasvit_" if nasvit_arch else ""
                        new_key = f'{nasvit_mark}transattn_{hw}_{cin}_{act}_{v}_{"ln" if layer_norm else "bn"}'
                        if new_key not in lut_result:
                            if lut_result_ref and new_key in lut_result_ref:
                                print(new_key)
                                continue
                            else:
                                build_models(new_key, "transattn", hw, cin, cout, exp, s, act, v=v, ds=ds, ds_exp=ds_exp)
                                lut_result[new_key] = [[hw, hw, cin]]
                        
                        # transffn
                        nasvit_mark = f"nasvit_" if nasvit_arch else ""
                        new_key = f'{nasvit_mark}transffn_{hw}_{cin}_{exp}_{act}_{"ln" if layer_norm else "bn"}'
                        if new_key not in lut_result:
                            if lut_result_ref and new_key in lut_result_ref:
                                print(new_key)
                                continue
                            else:
                                build_models(new_key, "transffn", hw, cin, cout, exp, s, act, v=v, ds=ds, ds_exp=ds_exp)
                                lut_result[new_key] = [[hw, hw, cin]]
    return lut_result


if __name__ == '__main__':

    lut_result = {}
    for configs in [search_space_config]:
        for i in range(9):
            if configs[i]["block_type"] == -1:
                lut_result = add_lut_key_firstconv(lut_result, configs[i],
                                            lut_result_ref=None, act=ACT)
                                            #   lut_result_ref="/data/data0/jiahang/nn-Meter/nn_meter/predictor/transformer_predictor/pixel4_lut.json")
            elif configs[i]["block_type"] == 0:
                lut_result = add_lut_key_conv(lut_result, configs[i],
                                            lut_result_ref=None, act=ACT)
                                            #   lut_result_ref="/data/data0/jiahang/nn-Meter/nn_meter/predictor/transformer_predictor/pixel4_lut.json")
            elif configs[i]["block_type"] == 1:
                stage_ds_se = False if i == 4 and nasvit_arch else True
                print(i, stage_ds_se)
                lut_result = add_lut_key_transformer(lut_result, configs[i],
                                                    lut_result_ref=None, stage_ds_se=stage_ds_se)
                                            #   lut_result_ref="/data/data0/jiahang/nn-Meter/nn_meter/predictor/transformer_predictor/pixel4_lut.json")
            elif configs[i]["block_type"] == 2:
                lut_result = add_lut_key_mbpool(lut_result, configs[i],
                                                    lut_result_ref=None, act=ACT)
                                            #   lut_result_ref="/data/data0/jiahang/nn-Meter/nn_meter/predictor/transformer_predictor/pixel4_lut.json")

        print(len(lut_result))

    res = {}
    for key, value in lut_result.items():
        if nasvit_arch:
            tmp_path = "/data1/jiahang/working/pixel6_supernet_workspace/predictor_build/kernels/nasvit_lut"
        else:
            if key.startswith("mbpool") or key.startswith("conv") or key.startswith("firstconv"):
                tmp_path = "/data1/jiahang/working/pixel6_supernet_workspace/predictor_build/kernels/common_layer"
            else:
                tmp_path = "/data1/jiahang/working/pixel6_supernet_workspace/predictor_build/kernels/trans_blocks"

        res[key] = {
            "converted_model": f"{tmp_path}/{key}.tflite",
            'shapes': value
        }
        # res[key] = 1

    with open(os.path.join(main_path, "results_pixel6", f"nasvit_lut_{'ln' if layer_norm else 'bn'}_v3.json"), 'w') as fp:
        json.dump({"lut": res}, fp, indent=4)
        # json.dump(res, fp, indent=4)


# nohup python /data/data0/jiahang/nn-Meter/examples/test_transformer/build_lut/build_lut_attnffn.py nasvit > nasvit_attnffn_lut_log.txt 2>&1 &
# [1] 42989
# nohup python /data/data0/jiahang/nn-Meter/examples/test_transformer/build_lut/build_lut_attnffn.py bn > trans_bn_lut_log.txt 2>&1 &
# 
# (base) jiahang@MSRAGPUM20:/data/data0/jiahang$ nohup python /data/data0/jiahang/nn-Meter/examples/test_transformer/build_lut/build_lut_attnffn.py nasvit > nasvit_attnffn_lut_log.txt 2>&1 &
# 
# (base) jiahang@MSRAGPUM20:/data/data0/jiahang$ nohup python /data/data0/jiahang/nn-Meter/examples/test_transformer/build_lut/build_lut_attnffn.py nasvit > nasvit_attnffn_lut_part3_log.txt 2>&1 &
# 
