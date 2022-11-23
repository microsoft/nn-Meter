import sys
import os, json
import tensorflow as tf
import tensorflow.keras as keras
from search_space.space_utils_large import configs as search_space_config
from search_space.space_utils_large import ACT as ACT
from keras_cv_attention_models.levit.levit import mhsa_with_multi_head_position_windows, res_mlp_block, mhsa_with_multi_head_position_windows_layer_norm, res_mlp_block_layer_norm
from keras_cv_attention_models.attention_layers import conv2d_no_bias, batchnorm_with_activation
sys.path.append("/home/v-chentang/vit/")
from fusednas_sampling2 import attention_downsampling, dsconv

mark = sys.argv[1]
layer_norm = True if mark == 'ln' else False
main_path = "/data/data0/jiahang/tflite_space/predictor_build/"
# print(mark, layer_norm)


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
        nn = mhsa_with_multi_head_position_windows(nn, channels, num_heads, key_dim, v_dim, 1, activation = ACT, name=name+f'layer_channel_'+str(channels)+str(layer_index))
        nn = res_mlp_block(nn, expansion_ratio, name=name+str(channels)+str(layer_index)+f'_ffn')
    else:
        nn = mhsa_with_multi_head_position_windows_layer_norm(nn, channels, num_heads, key_dim, v_dim, 1, activation=ACT, name=name+str(channels)+str(layer_index))
        nn = res_mlp_block_layer_norm(nn, expansion_ratio, name=name+str(channels)+str(layer_index)+f'_ffn')
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


def build_models(key, name, hw, cin, cout, exp, s, act, ks = None, v = None, ds = None, use_se = None, ds_exp = None):
    # return
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
        inputs = keras.Input(shape=[hw, hw, cin], batch_size=1)
        use_se = use_se or False
        output = conv_layer(inputs, cout, expansion_ratio=exp, stride=s, kernel_size=ks, act=act, use_se=use_se)

    elif name == "transformer": # transformer
        if ds == "ds":
            inputs = keras.Input(shape=[hw, hw, cin], batch_size=1)
            layer_index = 0
        elif ds == "nods":
            inputs = keras.Input(shape=[hw * hw, cin], batch_size=1)
            layer_index = 1
        output = transformer_layer(inputs, cout, expansion_ratio=exp, ds_exp=ds_exp, v_scale=v, stride=s, layer_index=layer_index, name=name, layer_norm=layer_norm)
    
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
    converted_model = os.path.join(main_path, "nasvit_lut", f"{key}.tflite")
    open(converted_model, 'wb').write(tflite_model)


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


def main_for_build_lut():
    lut_result = {}
    for configs in [search_space_config]:
        for i in range(9):
            if configs[i]["block_type"] == -1:
                lut_result = add_lut_key_firstconv(lut_result, configs[i],
                                            lut_result_ref=None)
                                            #   lut_result_ref="/data/data0/jiahang/nn-Meter/nn_meter/predictor/transformer_predictor/pixel4_lut.json")
            elif configs[i]["block_type"] == 0:
                lut_result = add_lut_key_conv(lut_result, configs[i],
                                            lut_result_ref=None)
                                            #   lut_result_ref="/data/data0/jiahang/nn-Meter/nn_meter/predictor/transformer_predictor/pixel4_lut.json")
            elif configs[i]["block_type"] == 1:
                lut_result = add_lut_key_transformer(lut_result, configs[i],
                                                    lut_result_ref=None)
                                            #   lut_result_ref="/data/data0/jiahang/nn-Meter/nn_meter/predictor/transformer_predictor/pixel4_lut.json")
            elif configs[i]["block_type"] == 2:
                lut_result = add_lut_key_mbpool(lut_result, configs[i],
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

    with open(os.path.join(main_path, "results_pixel6", f"lut_{'ln' if layer_norm else 'bn'}_v5.json"), 'w') as fp:
        json.dump({"lut": res}, fp, indent=4)
        # json.dump(res, fp, indent=4)


def main_for_build_model():
    downsampling = (False, True, True, True, True, False, True) # didn't contain the first conv3x3
    use_se = (False, False, True)

    sample = [
        160, 
        (24, 24, 40, 48, 64, 160, 320), 
        (1, 2, 3, 1, 6, 3, 1), 
        (1, 2, 2, 4, 4, 4), 
        (3, 3, 3, 3, 3, 3), 
        (4, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1), 
        (3, 4, 4, 4, 4, 4, 4, 10, 10, 10, 20), 
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), 
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), 
        (2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 4)
    ]
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
    # main_for_build_lut()
    main_for_build_model()
    
# nohup python /data/data0/jiahang/nn-Meter/examples/test_transformer/build_lut/build_lut.py ln > trans_ln_lut_log.txt 2>&1 &

# nohup python /data/data0/jiahang/nn-Meter/examples/test_transformer/build_lut/build_lut.py ln > trans_ln_lut_part2_log.txt 2>&1 &

# nohup python /data/data0/jiahang/nn-Meter/examples/test_transformer/build_lut/build_lut.py bn > trans_bn_lut_log.txt 2>&1 &
# [1] 29669

# scp /data/data0/jiahang/tflite_space/predictor_build/kernels/* jiahang@10.172.141.20:/data1/jiahang/working/pixel6_supernet_workspace/predictor_build/models/
# scp /data/data0/jiahang/tflite_space/predictor_build/results_pixel6/lut_v6.json jiahang@10.172.141.20:/data1/jiahang/working/pixel6_supernet_workspace/predictor_build/results/