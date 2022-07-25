import os, json, math
import tensorflow as tf
import tensorflow.keras as keras
from conv_block import conv_layer
from transformer_block import transformer_layer
from space_utils import ACT, configs

with open("/data/data0/jiahang/tflite_space/predictor_build/results/lut_v5.json", 'r') as fp:
    res = json.load(fp)

with open("/data/data0/jiahang/tflite_space/predictor_build/results/lut_filtered_key.json", 'r') as fp:
    refer_res = json.load(fp)

for k, v in res.items():
    if k in refer_res and refer_res[k] != -1:
        print(abs(res[k] - refer_res[k]) / res[k])
# def build_models(key, name, hw, cin, cout, exp, s, act, ks = None, v = None, ds = None):
#     return
#     if os.path.isfile(f"/data/data0/jiahang/tflite_space/predictor_build/kernels/{key}.tflite"):
#         return

#     print(key)
#     if name == "conv": # conv
#         inputs = keras.Input(shape=[hw, hw, cin], batch_size=1)
#         output = conv_layer(inputs, cout, expansion_ratio=exp, stride=s, kernel_size=ks, act=ACT)
#     else: # transformer
#         if ds == "ds":
#             inputs = keras.Input(shape=[hw, hw, cin], batch_size=1)
#             layer_index = 0
#         elif ds == "nods":
#             inputs = keras.Input(shape=[hw * hw, cin], batch_size=1)
#             layer_index = 1
#         output = transformer_layer(inputs, cout, expansion_ratio=exp, v_scale=v, stride=s, layer_index=layer_index, name=name)

#     model = keras.Model(inputs=inputs, outputs=output)
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     converter.target_spec.supported_ops = [
#         tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
#         tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
#     ]
#     tflite_model = converter.convert()
#     converted_model = f"/data/data0/jiahang/tflite_space/predictor_build/kernels/{key}.tflite"
#     open(converted_model, 'wb').write(tflite_model)


# def add_lut_key_conv(lut_result, config, hw_lis = None, lut_result_ref = None):
#     """
#     Args:
#         lut_result_ref (_type_, optional): only output keys not in lut_result_ref

#     Returns:
#         lut_result: dict of {block_config: 1}
#     """
#     name = "conv"

#     if hw_lis == None:
#         hw_lis = config["hw"]
#     for hw in hw_lis:
#         for cin in config["cin"]:
#             for cout in config["cout"]:
#                 for ks in config['kernel size']:
#                     for exp in config['expansion_ratio']:
#                         for act in [ACT]:
#                             for s in [1, 2]:
#                                 key = f'{name}_{hw}_{cin}_{cout}_{exp}_{s}_{act}_{ks}'
#                                 if key not in lut_result:
#                                     if lut_result_ref and key in lut_result_ref:
#                                         continue
#                                     else:
#                                         # build_models(key, name, hw, cin, cout, exp, s, act, ks=ks)
#                                         lut_result[key] = refer_res[key]
#                                         refer_res[key] = -1
#                                         # return lut_result
#     return lut_result


# def add_lut_key_transformer(lut_result, config, hw_lis = None, lut_result_ref = None):
#     """
#     Args:
#         lut_result_ref (_type_, optional): only output keys not in lut_result_ref

#     Returns:
#         lut_result: dict of {block_config: 1}
#     """
#     name = "transformer"

#     if hw_lis == None:
#         hw_lis = config["hw"]

#     ds = "ds"
#     for hw in hw_lis:
#         for cin in config["cin"][0]:
#             for cout in config["cout"]:
#                 for exp in config['expansion_ratio']:
#                     for act in [ACT]:
#                         for s in [config['stride']]:
#                             for v in config['v_scale']:
#                                 # layer_index = 0 / ds
#                                 new_key = f'{name}_{hw}_{cin}_{cout}_{exp}_{s}_{act}_{v}_{ds}'
#                                 old1k2e3y = f'{name}_{hw}_{cin}_{cout}_{exp}_{s}_{act}_{v}'
#                                 if old1k2e3y not in lut_result:
#                                     if lut_result_ref and new_key in lut_result_ref:
#                                         continue
#                                     else:
#                                         # build_models(key, name, hw, cin, cout, exp, s, act, v=v, ds=ds)
#                                         lut_result[new_key] = refer_res[old1k2e3y] if old1k2e3y in refer_res else 0
#                                         refer_res[old1k2e3y] = -1
#                                         # return lut_result

#     ds = "nods"
#     for hw in hw_lis:
#         for cin in config["cin"][1]:
#             for cout in config["cout"]:
#                 for exp in config['expansion_ratio']:
#                     for act in [ACT]:
#                         for s in [None]:
#                             for v in config['v_scale']:
#                                 new_key = f'{name}_{hw}_{cin}_{cout}_{exp}_{s}_{act}_{v}_{ds}'
#                                 old1k2e3y = f'{name}_{hw}_{cin}_{cout}_{exp}_1_{act}_{v}'
#                                 if old1k2e3y not in lut_result:
#                                     if lut_result_ref and new_key in lut_result_ref:
#                                         continue
#                                     elif cin != cout:
#                                         continue
#                                     else:
#                                         lut_result[new_key] = refer_res[old1k2e3y] if old1k2e3y in refer_res else 0
#                                         refer_res[old1k2e3y] = -1
#                                 # else:
#                                 #     print("!!!!!", old1k2e3y, lut_result[old1k2e3y])
#     return lut_result


# lut_result = {}
# for i in [0, 1, 2, 3]:
#     lut_result = add_lut_key_conv(lut_result, configs[i])

# for i in [4, 5, 6]:
#     lut_result = add_lut_key_transformer(lut_result, configs[i])
# print(len(lut_result))

# # res = {}
# # for key, value in lut_result.items():
# #     res[key] = {
# #         'model': f"/data/data0/jiahang/tflite_space/predictor_build/kernels/{key}.tflite",
# #         'shapes': value
# #     }

# with open("/data/data0/jiahang/tflite_space/predictor_build/results/lut_new_key.json", 'w') as fp:
#     json.dump(lut_result, fp, indent=4)

# with open("/data/data0/jiahang/tflite_space/predictor_build/results/lut_filtered_key.json", 'w') as fp:
#     json.dump(refer_res, fp, indent=4)
    
# # nohup /home/v-chentang/anaconda3/bin/python vit_lut/build_lut.py > trans_lut_log.txt 2>&1 &
# # [1] 6168