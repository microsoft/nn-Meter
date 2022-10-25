import sys
import os, json
import tensorflow as tf
import tensorflow.keras as keras
from search_space.space_utils_large import configs as search_space_config
from search_space.space_utils_large import ACT as ACT
from keras_cv_attention_models.levit.levit import mhsa_with_multi_head_position_windows, res_mlp_block, mhsa_with_multi_head_position_windows_layer_norm, res_mlp_block_layer_norm

sys.path.append("/home/v-chentang/vit/")
from fusednas_sampling2 import attention_downsampling, dsconv


def transformer_layer(inputs, channels, expansion_ratio, ds_exp, v_scale, stride, layer_index, name, layer_norm = False):
    if layer_index == 0:
        nn = attention_downsampling(inputs, channels, downsampling=stride==2, exp=ds_exp, use_se=True, dwconv=True)
    else:
        nn = inputs

    head_dim = 16
    num_heads = channels // head_dim # use fixed head dims here
    key_dim = head_dim
    v_dim = head_dim * v_scale

    if not layer_norm:
        nn = mhsa_with_multi_head_position_windows(nn, channels, num_heads, key_dim, v_dim, 1, activation=ACT, name=name+f'layer_channel_'+str(channels)+str(layer_index))
        nn = res_mlp_block(nn, expansion_ratio, name=name+str(channels)+str(layer_index)+f'_ffn')
    else:
        nn = mhsa_with_multi_head_position_windows_layer_norm(nn, channels, num_heads, key_dim, v_dim, 1, activation=ACT, name=name+str(channels)+str(layer_index))
        nn = res_mlp_block_layer_norm(nn, expansion_ratio, name=name+str(channels)+str(layer_index)+f'_ffn')
    return nn


def conv_layer(inputs, channel, expansion_ratio, kernel_size, stride, use_se, act=ACT):
    return dsconv(inputs, channel, strides=stride, kernel_size=kernel_size, exp=expansion_ratio, act=act, use_se=use_se)

mark = sys.argv[1]
layer_norm = True if mark == 'ln' else False
main_path = "/data/data0/jiahang/tflite_space/predictor_build/"
# print(mark, layer_norm)

def build_models(key, name, hw, cin, cout, exp, s, act, ks = None, v = None, ds = None, use_se = None, ds_exp = None):
    return
    print(key)
    if os.path.isfile(os.path.join(main_path, "kernels", f"{key}.tflite")):
        return        

    if name == "conv": # conv
        if layer_norm: return
        inputs = keras.Input(shape=[hw, hw, cin], batch_size=1)
        if use_se != None:
            output = conv_layer(inputs, cout, expansion_ratio=exp, stride=s, kernel_size=ks, act=act, use_se=use_se)
        else:
            output = conv_layer(inputs, cout, expansion_ratio=exp, stride=s, kernel_size=ks, act=act, use_se=False)
    else: # transformer
        if ds == "ds":
            inputs = keras.Input(shape=[hw, hw, cin], batch_size=1)
            layer_index = 0
        elif ds == "nods":
            inputs = keras.Input(shape=[hw * hw, cin], batch_size=1)
            layer_index = 1
        output = transformer_layer(inputs, cout, expansion_ratio=exp, ds_exp=ds_exp, v_scale=v, stride=s, layer_index=layer_index, name=name, layer_norm=layer_norm)

    model = keras.Model(inputs=inputs, outputs=output)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()
    converted_model = os.path.join(main_path, "kernels", f"{key}.tflite")
    open(converted_model, 'wb').write(tflite_model)
    # keras.models.save_model(model, f"/data/data0/jiahang/tflite_space/predictor_build/kernels/{key}")


def add_lut_key_conv(lut_result, config, hw_lis = None, lut_result_ref = None):
    """
    Args:
        lut_result_ref (_type_, optional): only output keys not in lut_result_ref

    Returns:
        lut_result: dict of {block_config: 1}
    """
    if layer_norm:
        return lut_result

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
                        for act in [ACT]:
                            try:
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


def add_lut_key_transformer(lut_result, config, hw_lis = None, lut_result_ref = None):
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
                                    new_key = f'{name}_{hw}_{cin}_{cout}_{exp}_{s}_{act}_{v}_{ds}_{ds_exp}_{"ln" if layer_norm else "bn"}'
                                    if new_key not in lut_result:
                                        if lut_result_ref and new_key in lut_result_ref:
                                            print(new_key)
                                            continue
                                        else:
                                            build_models(new_key, name, hw, cin, cout, exp, s, act, v=v, ds=ds, ds_exp=ds_exp)
                                            lut_result[new_key] = [[hw, hw, cin]]
                                            # return lut_result

    ds = "nods"
    for hw in config["hw_out"]:
        for cout in config["channel"]:
            cin = cout
            for exp in config['expansion_ratio']:
                for act in [ACT]:
                    for v in config['v_scale']:
                        new_key = f'{name}_{hw}_{cin}_{cout}_{exp}_1_{act}_{v}_{ds}_{"ln" if layer_norm else "bn"}' #TODO
                        if new_key not in lut_result:
                            if lut_result_ref and new_key in lut_result_ref:
                                print(new_key)
                                continue
                            elif cin != cout:
                                continue
                            else:
                                build_models(new_key, name, hw, cin, cout, exp, 1, act, v=v, ds=ds)
                                lut_result[new_key] = [[hw * hw, cin]]
                                # return lut_result
    return lut_result


lut_result = {}
for configs in [search_space_config]:
    for i in range(1, 8):
        if configs[i]["block_type"] == 0:
            lut_result = add_lut_key_conv(lut_result, configs[i],
                                          lut_result_ref=None)
                                        #   lut_result_ref="/data/data0/jiahang/nn-Meter/nn_meter/predictor/transformer_predictor/pixel4_lut.json")
        else:
            lut_result = add_lut_key_transformer(lut_result, configs[i],
                                                 lut_result_ref=None)
                                        #   lut_result_ref="/data/data0/jiahang/nn-Meter/nn_meter/predictor/transformer_predictor/pixel4_lut.json")
print(len(lut_result))

res = {}
for key, value in lut_result.items():
    res[key] = {
        "converted_model": f"/data1/jiahang/working/pixel6_supernet_workspace/predictor_build/kernels/first90000/{key}.tflite",
        'shapes': value
    }
    # res[key] = 1

with open(os.path.join(main_path, "results_pixel6", f"lut_{'ln' if layer_norm else 'bn'}_v2.json"), 'w') as fp:
    json.dump({"lut": res}, fp, indent=4)
    # json.dump(res, fp, indent=4)


# nohup python /data/data0/jiahang/nn-Meter/examples/test_transformer/build_lut/build_lut.py ln > trans_ln_lut_log.txt 2>&1 &
# [1] 52544
# nohup python /data/data0/jiahang/nn-Meter/examples/test_transformer/build_lut/build_lut.py bn > trans_bn_lut_log.txt 2>&1 &
# [2] 54749

# scp /data/data0/jiahang/tflite_space/predictor_build/kernels/* jiahang@10.172.141.20:/data1/jiahang/working/pixel6_supernet_workspace/predictor_build/models/
# scp /data/data0/jiahang/tflite_space/predictor_build/results_pixel6/lut_v6.json jiahang@10.172.141.20:/data1/jiahang/working/pixel6_supernet_workspace/predictor_build/results/