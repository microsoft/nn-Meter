import os, json
import tensorflow as tf
import tensorflow.keras as keras
from conv_block import conv_layer
from transformer_block import transformer_layer

ACT = 'hard_swish'
main_path = "/data/data0/jiahang/tflite_space/predictor_build/"

def build_models(key, name, hw, cin, cout, exp, s, act, ks = None, v = None, ds = None, use_se = None):
    # return
    if os.path.isfile(os.path.join(main_path, "kernels", f"{key}.tflite")):
        return        

    print(key)
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
        output = transformer_layer(inputs, cout, expansion_ratio=exp, v_scale=v, stride=s, layer_index=layer_index, name=name)

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
    if lut_result_ref:
        with open(os.path.join(main_path, "results", lut_result_ref), 'r') as fp:
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
                                new_key = f'{name}_{hw}_{cin}_{cout}_{exp}_{s}_{act}_{v}_{ds}'
                                if new_key not in lut_result:
                                    if lut_result_ref and new_key in lut_result_ref:
                                        continue
                                    else:
                                        build_models(new_key, name, hw, cin, cout, exp, s, act, v=v, ds=ds)
                                        lut_result[new_key] = [[hw, hw, cin]]
                                        # return lut_result

    ds = "nods"
    for hw in config["hw_out"]:
        for cout in config["channel"]:
            cin = cout
            for exp in config['expansion_ratio']:
                for act in [ACT]:
                    for v in config['v_scale']:
                        new_key = f'{name}_{hw}_{cin}_{cout}_{exp}_1_{act}_{v}_{ds}' #TODO
                        if new_key not in lut_result:
                            if lut_result_ref and new_key in lut_result_ref:
                                continue
                            elif cin != cout:
                                continue
                            else:
                                build_models(new_key, name, hw, cin, cout, exp, 1, act, v=v, ds=ds)
                                lut_result[new_key] = [[hw * hw, cin]]
                                # return lut_result
    return lut_result


lut_result = {}

from space_utils import configs as c3t3_config
from space_utils_2c4t import configs as c2t4_config
from space_utils_4c2t import configs as c4t2_config
from space_utils_attnnas import configs as attnnas_config
from space_utils_spacev1 import configs as spacev1_config
# for configs in [c3t3_config, c2t4_config, c4t2_config]:
for configs in [attnnas_config, spacev1_config]:
    for i in range(1, 7):
        if configs[i]["block_type"] == 0:
            lut_result = add_lut_key_conv(lut_result, configs[i], "lut_v5_all.json")
        else:
            lut_result = add_lut_key_transformer(lut_result, configs[i], "lut_v5_all.json")
print(len(lut_result))

res = {}
for key, value in lut_result.items():
    res[key] = {
        "converted_model": f"/data1/jiahang/working/pixel6_supernet_workspace/predictor_build/models/{key}.tflite",
        'shapes': value
    }
    # res[key] = 1

with open(os.path.join(main_path, "results_pixel6", "lut_v6.json"), 'w') as fp:
    json.dump({"lut": res}, fp, indent=4)
    # json.dump(res, fp, indent=4)

# nohup /home/v-chentang/anaconda3/bin/python vit_lut/build_lut.py > trans_lut_log.txt 2>&1 &
# [1] 10136

# scp /data/data0/jiahang/tflite_space/predictor_build/kernels/* jiahang@10.172.141.20:/data1/jiahang/working/pixel6_supernet_workspace/predictor_build/models/
# scp /data/data0/jiahang/tflite_space/predictor_build/results_pixel6/lut_v6.json jiahang@10.172.141.20:/data1/jiahang/working/pixel6_supernet_workspace/predictor_build/results/