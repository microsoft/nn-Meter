import sys
import os, json
import argparse
import tensorflow as tf
import tensorflow.keras as keras
from search_space.space_utils_large_v2 import configs as search_space_config
from search_space.space_utils_large_v2 import ACT as ACT
# from search_space.space_utils_nasvit import configs as search_space_config
# from search_space.space_utils_nasvit import ACT as ACT
from build_lut_modules import (conv_layer, first_conv_layer, mbpool_layer, 
                               transformer_ds, transformer_attn, transformer_ffn, transformer_layer,
                               nasvit_transformer_ds, nasvit_transformer_attn, nasvit_transformer_ffn, nasvit_transformer_layer)

parser = argparse.ArgumentParser()
parser.add_argument('--mark', default='ln', type=str)
parser.add_argument('--lut-mode', default='layer_mode', type=str)
# parser.add_argument('--build-model', type=str, nargs='+')
parser.add_argument('--lut-refer', default=None, type=str)
args = parser.parse_args()

mark = args.mark
layer_norm = True if mark == 'ln' or mark == 'nasvit' else False
nasvit_arch = True if mark == 'nasvit' else False
ACT = 'swish' if nasvit_arch else ACT
main_path = "/data/data0/jiahang/tflite_space/predictor_build/"

lut_mode = args.lut_mode # lut_model: block_mode or layer_mode
# print(mark, layer_norm)


def build_models(key, name, hw, cin, cout, exp, s, act, ks = None, v = None, ds = None, use_se = None, ds_exp = None):
    # return
    
    if os.path.isfile(os.path.join(main_path, "common_layer", f"{key}.tflite")):
        return
    if os.path.isfile(os.path.join(main_path, "trans_blocks", f"{key}.tflite")):
        return
    if os.path.isfile(os.path.join(main_path, "nasvit_layer", f"{key}.tflite")):
        return
    if os.path.isfile(os.path.join(main_path, "trans_layer", f"{key}.tflite")):
        return

    if name == "firstconv":
        inputs = keras.Input(shape=[hw, hw, cin], batch_size=1)
        output = first_conv_layer(inputs, cout, stride=s, kernel_size=ks, act=act)

    if name == "conv": # conv
        inputs = keras.Input(shape=[hw, hw, cin], batch_size=1)
        use_se = use_se or False
        output = conv_layer(inputs, cout, expansion_ratio=exp, stride=s, kernel_size=ks, act=act, use_se=use_se)
    
    elif name == "transds":
        se = use_se if nasvit_arch else True
        cls = nasvit_transformer_ds if nasvit_arch else transformer_ds
        inputs = keras.Input(shape=[hw * hw, cin], batch_size=1)
        output = cls(inputs, cout, ds_exp=ds_exp, stride=s, act=act, se=se)
    
    elif name == "transattn": # transformer attention layer
        cls = nasvit_transformer_attn if nasvit_arch else transformer_attn
        inputs = keras.Input(shape=[hw * hw, cin], batch_size=1)
        output = cls(inputs, cout, v_scale=v, name=name, layer_norm=layer_norm, act=act)
    
    elif name == "transffn": # transformer attention layer
        cls = nasvit_transformer_ffn if nasvit_arch else transformer_ffn
        inputs = keras.Input(shape=[hw * hw, cin], batch_size=1)
        output = cls(inputs, cout, expansion_ratio=exp, name=name, layer_norm=layer_norm, act=act)

    elif name == "transformer": # transformer
        se = use_se if nasvit_arch else True
        cls = nasvit_transformer_layer if nasvit_arch else transformer_layer
        if ds == "ds":
            inputs = keras.Input(shape=[hw, hw, cin], batch_size=1)
            layer_index = 0
        elif ds == "nods":
            inputs = keras.Input(shape=[hw * hw, cin], batch_size=1)
            layer_index = 1
        output = cls(inputs, cout, expansion_ratio=exp, ds_exp=ds_exp, v_scale=v, stride=s, layer_index=layer_index, name=name, layer_norm=layer_norm, act=act, se=se)

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
    dir_name = "nasvit_layer" if nasvit_arch else "trans_block_new"
    converted_model = os.path.join(main_path, dir_name, f"{key}.tflite")
    open(converted_model, 'wb').write(tflite_model)


def add_lut_key_transformer(lut_result, config, hw_lis = None, lut_result_ref = None, stage_ds_se = True):
    """
    Args:
        lut_result_ref (_type_, optional): only output keys not in lut_result_ref

    Returns:
        lut_result: dict of {block_config: 1}
    """
    name = "transformer"
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
                                    
                                    if lut_mode == "layer_mode":
                                        nasvit_mark = f"nasvit_{'se' if stage_ds_se else 'nose'}_" if nasvit_arch else ""
                                        new_key = f'{nasvit_mark}{name}_{hw}_{cin}_{cout}_{exp}_{s}_{act}_{v}_{ds}_{ds_exp}_{"ln" if layer_norm else "bn"}'
                                        if new_key not in lut_result:
                                            if lut_result_ref and new_key in lut_result_ref:
                                                print(new_key)
                                                continue
                                            else:
                                                build_models(new_key, name, hw, cin, cout, exp, s, act, v=v, ds=ds, ds_exp=ds_exp)
                                                lut_result[new_key] = [[hw, hw, cin]]
                                                # return lut_result
                                    
                                    elif lut_mode == "block_mode":
                                        # transds
                                        nasvit_mark = f"nasvit_{'se' if stage_ds_se else 'nose'}_" if nasvit_arch else ""
                                        new_key = f'{nasvit_mark}transds_{hw}_{cin}_{cout}_{s}_{ds_exp}'
                                        if new_key not in lut_result:
                                            if lut_result_ref and new_key in lut_result_ref:
                                                print(new_key)
                                                pass
                                            else:
                                                build_models(new_key, "transds", hw, cin, cout, exp, s, act, use_se=stage_ds_se, v=v, ds=ds, ds_exp=ds_exp)
                                                lut_result[new_key] = [[hw, hw, cin]]

                                        # transattn
                                        nasvit_mark = f"nasvit_" if nasvit_arch else ""
                                        new_key = f'{nasvit_mark}transattn_{hw}_{cout}_{act}_{v}_{"ln" if layer_norm else "bn"}'
                                        if new_key not in lut_result:
                                            if lut_result_ref and new_key in lut_result_ref:
                                                print(new_key)
                                                pass
                                            else:
                                                build_models(new_key, "transattn", hw, cout, cout, exp, s, act, v=v, ds=ds, ds_exp=ds_exp)
                                                lut_result[new_key] = [[hw, hw, cin]]
                                        
                                        # transffn
                                        nasvit_mark = f"nasvit_" if nasvit_arch else ""
                                        new_key = f'{nasvit_mark}transffn_{hw}_{cout}_{exp}_{act}_{"ln" if layer_norm else "bn"}'
                                        if new_key not in lut_result:
                                            if lut_result_ref and new_key in lut_result_ref:
                                                print(new_key)
                                                pass
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
                        if lut_mode == "layer_mode":
                            nasvit_mark = f"nasvit_{'se' if stage_ds_se else 'nose'}_" if nasvit_arch else ""
                            new_key = f'{nasvit_mark}{name}_{hw}_{cin}_{cout}_{exp}_1_{act}_{v}_{ds}_{"ln" if layer_norm else "bn"}' #TODO
                            if new_key not in lut_result:
                                if lut_result_ref and new_key in lut_result_ref:
                                    print(new_key)
                                    pass
                                elif cin != cout:
                                    pass
                                else:
                                    build_models(new_key, name, hw, cin, cout, exp, 1, act, v=v, ds=ds)
                                    lut_result[new_key] = [[hw * hw, cin]]
                                    # return lut_result
                        
                        elif lut_mode == "block_mode":
                            # transattn
                            nasvit_mark = f"nasvit_" if nasvit_arch else ""
                            new_key = f'{nasvit_mark}transattn_{hw}_{cin}_{act}_{v}_{"ln" if layer_norm else "bn"}'
                            if new_key not in lut_result:
                                if lut_result_ref and new_key in lut_result_ref:
                                    print(new_key)
                                    pass
                                else:
                                    build_models(new_key, "transattn", hw, cin, cout, exp, s, act, v=v, ds=ds, ds_exp=ds_exp)
                                    lut_result[new_key] = [[hw, hw, cin]]
                            
                            # transffn
                            nasvit_mark = f"nasvit_" if nasvit_arch else ""
                            new_key = f'{nasvit_mark}transffn_{hw}_{cin}_{exp}_{act}_{"ln" if layer_norm else "bn"}'
                            if new_key not in lut_result:
                                if lut_result_ref and new_key in lut_result_ref:
                                    print(new_key)
                                    pass
                                else:
                                    build_models(new_key, "transffn", hw, cin, cout, exp, s, act, v=v, ds=ds, ds_exp=ds_exp)
                                    lut_result[new_key] = [[hw, hw, cin]]
    return lut_result


def add_lut_key_conv(lut_result, config, hw_lis = None, lut_result_ref = None, act = ACT):
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

    name = "conv"

    # layer_index = 0
    for hw in config["hw"]:
        for cin in config["cin"]:
            for cout in config["channel"]:
                for ks in config['kernel size']:
                    for exp in config['expansion_ratio']:
                        for act in [act]:
                            # try:
                                for s in [config['stride']]:
                                    se = config["use_se"] # TODO: change 'True' or 'False' to 'se' and 'nose'
                                    key = f'{name}_{hw}_{cin}_{cout}_{exp}_{s}_{act}_{ks}_{"se" if se else "nose"}'
                                    if key not in lut_result:
                                        if lut_result_ref and key in lut_result_ref:
                                            print(key)
                                            continue
                                        else:
                                            build_models(key, name, hw, cin, cout, exp, s, act, ks=ks, use_se=se)
                                            lut_result[key] = [[hw, hw, cin]]
                                            # return lut_result
                            # except:
                            #     import pdb; pdb.set_trace()
    # layer_index > 0
    if config['name'] == 'first_conv':
        return lut_result
    for hw in config["hw_out"]:
        for cin in config["channel"]:
            cout = cin
            for ks in config['kernel size']:
                for exp in config['expansion_ratio']:
                    for act in [act]:
                        
                        se = config["use_se"] # TODO: change 'True' or 'False' to 'se' and 'nose'
                        key = f'{name}_{hw}_{cin}_{cout}_{exp}_1_{act}_{ks}_{"se" if se else "nose"}'
                        
                        if key not in lut_result:
                            if lut_result_ref and key in lut_result_ref:
                                print(key)
                                continue
                            else:
                                build_models(key, name, hw, cin, cout, exp, s, act, ks=ks, use_se=se)
                                lut_result[key] = [[hw, hw, cin]]
                                # return lut_result
    return lut_result


def add_lut_key_mbpool(lut_result, config, hw_lis = None, lut_result_ref = None, act = ACT):
    name = "mbpool"
    if lut_result_ref:
        with open(lut_result_ref, 'r') as fp:
            lut_result_ref = json.load(fp)
    else:
        lut_result_ref = None
    
    for hw in config["hw"]:
        for cin in config["cin"]:
            for cout in config["channel"]:
                for exp in config['expansion_ratio']:
                    for act in [act]:
                        new_key = f'{name}_{hw}_{cin}_{cout}_{exp}_{act}'
                        if new_key not in lut_result:
                            if lut_result_ref and new_key in lut_result_ref:
                                print(new_key)
                                continue
                            else:
                                build_models(new_key, name, hw, cin, cout, exp, None, act, v=None, ds=None)
                                lut_result[new_key] = [[hw * hw, cin]]
                                # return lut_result
    return lut_result


def add_lut_key_firstconv(lut_result, config, hw_lis = None, lut_result_ref = None, act = ACT):
    name = "firstconv"
    if lut_result_ref:
        with open(lut_result_ref, 'r') as fp:
            lut_result_ref = json.load(fp)
    else:
        lut_result_ref = None
    
    # layer_index = 0
    for hw in config["hw"]:
        for cin in config["cin"]:
            for cout in config["channel"]:
                for ks in config['kernel size']:
                    for act in [act]:
                        for s in [config['stride']]:
                            nasvit_mark = "_swish" if act == "swish" else ""
                            key = f'{name}_{hw}_{cin}_{cout}_{s}_{ks}{nasvit_mark}'
                            if key not in lut_result:
                                if lut_result_ref and key in lut_result_ref:
                                    print(key)
                                    continue
                                else:
                                    build_models(key, name, hw, cin, cout, None, s, act, ks=ks)
                                    lut_result[key] = [[hw, hw, cin]]
    return lut_result


def main_for_build_lut():
    lut_result = {}
    for configs in [search_space_config]:
        for i in range(9):
            if configs[i]["block_type"] == -1:
                lut_result = add_lut_key_firstconv(lut_result, configs[i],
                                            lut_result_ref=args.lut_refer, act=ACT)
            elif configs[i]["block_type"] == 0:
                lut_result = add_lut_key_conv(lut_result, configs[i],
                                            lut_result_ref=args.lut_refer, act=ACT)
            elif configs[i]["block_type"] == 1:
                stage_ds_se = False if i == 4 and nasvit_arch else True
                configs[i]['downsample_expansion_ratio'] = [4, 6]
                lut_result = add_lut_key_transformer(lut_result, configs[i],
                                                    lut_result_ref=args.lut_refer, stage_ds_se=stage_ds_se)
            elif configs[i]["block_type"] == 2:
                lut_result = add_lut_key_mbpool(lut_result, configs[i],
                                                    lut_result_ref=args.lut_refer, act=ACT)

    print(len(lut_result))

    res = {}
    for key, value in lut_result.items():
        if nasvit_arch:
            tmp_path = "/data1/jiahang/working/pixel6_supernet_workspace/predictor_build/kernels/nasvit_layer"
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

    output_path = os.path.join(main_path, "results_pixel6", f"{'nasvit_' if nasvit_arch else ''}lut_{'ln' if layer_norm else 'bn'}_{lut_mode}_v8.json")
    with open(output_path, 'w') as fp:
        json.dump({"lut": res}, fp, indent=4)
        # json.dump(res, fp, indent=4)


def build_model_by_config(inputs, channels, depths, conv_ratio, kr_size, mlp_ratio,
                          num_heads, window_size, qk_scale, v_scale, downsampling,
                          nasvit_arch=False, se=None, stage=['C', 'C', 'C', 'T', 'T', 'T', 'T'],
                          num_mlp=1, reproduce_nasvit=False):
    num_C = stage.count('C')
    
    nn = first_conv_layer(inputs, 16, act = ACT)

    for stage in range(len(channels)):
        for layer in range(depths[stage]):
            if stage < num_C:
                kr_exp_idx = sum([depths[i] for i in range(stage)]) + layer
                nn = conv_layer(nn, channels[stage], conv_ratio[kr_exp_idx], kr_size[kr_exp_idx], 2 if layer == 0 and downsampling[stage] else 1, se[stage], act = ACT)
            
            if stage >= num_C:
                tranformer_idx = sum([depths[i] for i in range(num_C, stage)]) + layer
                nn = transformer_layer(nn, channels[stage], mlp_ratio[tranformer_idx], 6, v_scale[tranformer_idx], 2 if layer == 0 and downsampling[stage] else 1, layer, f"trans{stage}{layer}", layer_norm = layer_norm)

    nn = mbpool_layer(nn, 1984, 6, act = ACT)
    model = tf.keras.models.Model(inputs, nn)
    return model


def build_nasvit_model_by_config(inputs, channels, depths, conv_ratio, kr_size, mlp_ratio,
                          num_heads, window_size, qk_scale, v_scale, downsampling,
                          nasvit_arch=False, se=None, stage=['C', 'C', 'C', 'T', 'T', 'T', 'T'],
                          num_mlp=1, reproduce_nasvit=False):
    num_C = stage.count('C')
    
    nn = first_conv_layer(inputs, 16, act = ACT)

    for stage in range(len(channels)):
        for layer in range(depths[stage]):
            if stage < num_C:
                kr_exp_idx = sum([depths[i] for i in range(stage)]) + layer
                nn = conv_layer(nn, channels[stage], conv_ratio[kr_exp_idx], kr_size[kr_exp_idx], 2 if layer == 0 and downsampling[stage] else 1, se[stage], act = ACT)
            
            if stage >= num_C:
                tranformer_idx = sum([depths[i] for i in range(num_C, stage)]) + layer
                nn = transformer_layer(nn, channels[stage], mlp_ratio[tranformer_idx], 6, v_scale[tranformer_idx], 2 if layer == 0 and downsampling[stage] else 1, layer, f"trans{stage}{layer}", layer_norm = layer_norm)

    nn = mbpool_layer(nn, 1984, 6, act = ACT)
    model = tf.keras.models.Model(inputs, nn)
    return model


def main_for_build_model(sample, mode = "ours"):
    downsampling = (False, True, True, True, True, False, True) # didn't contain the first conv3x3
    use_se = (False, False, True)

    res, channels, depths, conv_ratio, kr_size, mlp_ratio, num_heads, window_size, qk_scale, v_scale = sample
    inputs = tf.keras.layers.Input((res, res, 3))
    model = build_model_by_config(inputs, channels, depths, conv_ratio, kr_size, mlp_ratio,
                          num_heads, window_size, qk_scale, v_scale,
                          downsampling=downsampling, se=use_se)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()
    converted_model = os.path.join(main_path, f"test.tflite")
    open(converted_model, 'wb').write(tflite_model)


if __name__ == '__main__':
    main_for_build_lut()
    # main_for_build_model()


# nohup python /data/data0/jiahang/nn-Meter/examples/test_transformer/build_lut/build_lut.py --mark ln --lut-mode block_mode --lut-refer /data/data0/jiahang/nn-Meter/nn_meter/predictor/transformer_predictor/lut/pixel6_lut_ln_v2.json > our_space_v2_lut_log.txt 2>&1 &
# 46498
# nohup python /data/data0/jiahang/nn-Meter/examples/test_transformer/build_lut/build_lut.py nasvit block_mode > nasvit_block_lut_log.txt 2>&1 &
# 


# scp -r /data/data0/jiahang/tflite_space/predictor_build/nasvit_layer jiahang@10.172.141.68:/data1/jiahang/working/pixel6_supernet_workspace/predictor_build/models/
# scp /data/data0/jiahang/tflite_space/predictor_build/results_pixel6/lut_v6.json jiahang@10.172.141.68:/data1/jiahang/working/pixel6_supernet_workspace/predictor_build/results/
