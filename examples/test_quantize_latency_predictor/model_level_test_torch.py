from curses import KEY_SAVE
import logging
import os
import json
import time
import torch
from torch import nn
# from tensorflow import keras
from nn_meter.dataset.bench_dataset import latency_metrics
from nn_meter.builder.backends import connect_backend
from nn_meter.predictor import load_latency_predictor
from nn_meter.builder import builder_config
from .nas_models.networks.torch.mobilenetv3 import MobileNetV3Net
from .nas_models.blocks.torch.mobilenetv3_block import SE
from nn_meter.builder.nn_generator.torch_networks.utils import get_inputs_by_shapes
from .nas_models.blocks.torch.mobilenetv3_block import block_dict, BasicBlock
from .nas_models.search_space.mobilenetv3_space import MobileNetV3Space
from .nas_models.common import parse_sample_str


workspace = "/sdc/jiahang/working/ort_mobilenetv3_workspace"
builder_config.init(workspace)
backend = connect_backend(backend_name='ort_cpu_int8')
predictor_name = "onnxruntime_int8"
predictor = load_latency_predictor(predictor_name)


def profile_and_predict(model, input_shape, mark=""):
    print("\n")
    print(model)
    # input_shape example [3, 224, 224]
    torch.onnx.export(
            model,
            get_inputs_by_shapes([[*input_shape]], 1),
            f"/sdc/jiahang/working/ort_mobilenetv3_workspace/code/test.onnx",
            input_names=['input'],
            output_names=['output'],
            verbose=False,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
        )
    res = backend.profile_model_file(f"/sdc/jiahang/working/ort_mobilenetv3_workspace/code/test.onnx",
                               f"/sdc/jiahang/working/ort_mobilenetv3_workspace/code/",
                               input_shape=[[*input_shape]]
                            )    
    pred_lat = predictor.predict(model, "torch", input_shape=tuple([1] + input_shape), apply_nni=False) # in unit of ms
    print(f"[{mark}]: ", "profiled: ", res["latency"].avg, "predicted: ", pred_lat)
    input_shape = list(model(get_inputs_by_shapes([[*input_shape]], 1)).shape)[1:]
    return res["latency"].avg, pred_lat

## ------------- model level
sample_str = "ks55355773757755735757_e66643464363346436436_d22343"
# model = MobileNetV3Net(sample_str)
# real, pred = profile_and_predict(model, [3, 224, 224], mark="")


## ------------- block level
# width_mult = 1.0
# num_classes=1000
# block_dict=block_dict
# hw = 224
# space = MobileNetV3Space(width_mult=width_mult, num_classes=num_classes, hw=hw)

# sample_config = parse_sample_str(sample_str)

# blocks = []
# first_conv = block_dict['first_conv'](hwin=hw, cin=3, cout=space.stage_width[0])
# first_mbconv = block_dict['first_mbconv'](
#     hwin=hw//2,
#     cin=space.stage_width[0],
#     cout=space.stage_width[1]
# )
# blocks.append(first_conv)
# blocks.append(first_mbconv)

# hwin = hw // 2
# cin = space.stage_width[1]
# block_idx = 0
# for strides, cout, max_depth, depth, act, se in zip(
#     space.stride_stages[1:], space.stage_width[2:], 
#     space.num_block_stages[1:], sample_config['d'],
#     space.act_stages[1:], space.se_stages[1:]
# ):
#     for i in range(depth):
#         k = sample_config['ks'][block_idx + i]
#         e = sample_config['e'][block_idx + i]
#         strides = 1 if i > 0 else strides
#         print(hwin, cin, cout, k, strides)
#         blocks.append(block_dict['mbconv'](hwin, cin, cout, kernel_size=k, expand_ratio=e, strides=strides,
#             act=act, se=int(se)))
#         cin = cout 
#         hwin //= strides
#     block_idx += max_depth
# # blocks = nn.Sequential(*blocks)

# final_expand = block_dict['final_expand'].build_from_config(space.block_configs[-3])
# blocks.append(final_expand)
# feature_mix = block_dict['feature_mix'].build_from_config(space.block_configs[-2])
# blocks.append(feature_mix)
# logits = block_dict['logits'].build_from_config(space.block_configs[-1])
# blocks.append(logits)

# res_collection = []
# input_shape = [3, 224, 224]
# for i, block in enumerate(blocks):
#     real, pred = profile_and_predict(block, input_shape, mark=str(i))      
#     input_shape = list(block(get_inputs_by_shapes([[*input_shape]], 1)).shape)[1:]
#     res_collection.append([i, input_shape, real, pred])
#     # break
# print(res_collection)


## ------------- op level
from nn_meter.builder.nn_generator.torch_networks.blocks import ConvBnRelu, DwConvBnRelu, HswishBlock, SEBlock

# conv-bn-relu
reals, preds = [], []
configs = [
    [224, 3, 16, 3, 2], 
    [112, 16, 16, 1, 1], 
    [112, 16, 96, 1, 1], 
    [56, 96, 24, 1, 1], 
    [56, 24, 144, 1, 1], 
    [56, 144, 24, 1, 1], 
    [56, 24, 72, 1, 1], 
    [28, 72, 40, 1, 1], 
    [28, 40, 160, 1, 1], 
    [28, 160, 40, 1, 1], 
    [28, 40, 120, 1, 1], 
    [14, 120, 80, 1, 1], 
    [14, 80, 480, 1, 1], 
    [14, 480, 80, 1, 1], 
    [14, 80, 240, 1, 1], 
    [14, 240, 80, 1, 1], 
    [14, 80, 320, 1, 1], 
    [14, 320, 112, 1, 1], 
    [14, 112, 672, 1, 1], 
    [14, 672, 112, 1, 1], 
    [14, 112, 448, 1, 1], 
    [14, 448, 112, 1, 1], 
    [14, 112, 336, 1, 1], 
    [14, 336, 112, 1, 1], 
    [14, 112, 672, 1, 1], 
    [7, 672, 160, 1, 1], 
    [7, 160, 640, 1, 1], 
    [7, 640, 160, 1, 1], 
    [7, 160, 480, 1, 1], 
    [7, 480, 160, 1, 1], 
    [7, 160, 960, 1, 1], 
    [1, 960, 1280, 1, 1]
]
for i, config in enumerate(configs):
    hwin, cin, cout, k, strides = config
    config_in = {
        "HW": hwin,
        "CIN": cin,
        "COUT": cin,
        "KERNEL_SIZE": k,
        "STRIDES": strides
    }
    input_shape = [cin, hwin, hwin]
    model = ConvBnRelu(config_in).get_model()
    try:
        real, pred = profile_and_predict(model, input_shape, mark=str(i))
        reals.append(real)
        preds.append(pred)
    except:
        print("wrong!!", config)
        
rmse, rmspe, error, acc5, acc10, acc15 = latency_metrics(preds, reals)
for item in zip(preds, reals):
    print(item)
print(f"[Conv-bn-relu] rmse: {rmse}, rmspe: {rmspe}, error: {error}, acc5: {acc5}, acc10: {acc10}, acc15: {acc15}")


# hswish
class HSwish(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.relu6 = nn.ReLU(6)

    def forward(self, x):
        return x * self.relu6(x + 3.) * (1. / 6.)


reals, preds = [], []
configs = [
    [112, 16],
    [28, 120],
    [14, 120],
    [14, 480],
    [14, 480],
    [14, 240],
    [14, 240],
    [14, 320],
    [14, 320],
    [14, 672],
    [14, 672],
    [14, 448],
    [14, 448],
    [14, 336],
    [14, 336],
    [14, 672],
    [7, 672],
    [7, 640],
    [7, 640],
    [7, 480],
    [7, 480],
    [7, 960],
    [1, 1280],
]

for i, config in enumerate(configs):
    hwin, cin = config
    config_in = {
        "HW": hwin,
        "CIN": cin
    }
    input_shape = [cin, hwin, hwin]
    model = HSwish()
    # try:
    real, pred = profile_and_predict(model, input_shape, mark='')
    reals.append(real)
    preds.append(pred)
    # except:
        # print("wrong!!", config)
        
rmse, rmspe, error, acc5, acc10, acc15 = latency_metrics(preds, reals)
for item in zip(preds, reals):
    print(item)
print(f"[Hswish] rmse: {rmse}, rmspe: {rmspe}, error: {error}, acc5: {acc5}, acc10: {acc10}, acc15: {acc15}")



# dwconv-bn-relu
reals, preds = [], []
configs = [
    [112, 16, 16, 3, 1],
    [112, 96, 96, 5, 2],
    [56, 144, 144, 5, 1],
    [56, 72, 72, 5, 2],
    [28, 160, 160, 7, 1],
    [28, 120, 120, 7, 2],
    [14, 480, 480, 5, 1],
    [14, 240, 240, 7, 1],
    [14, 320, 320, 5, 1],
    [14, 672, 672, 5, 1],
    [14, 448, 448, 7, 1],
    [14, 336, 336, 3, 1],
    [14, 672, 672, 5, 2],
    [7, 640, 640, 7, 1],
    [7, 480, 480, 5, 1]
]
for i, config in enumerate(configs):
    hwin, cin, cout, k, strides = config
    config_in = {
        "HW": hwin,
        "CIN": cin,
        "COUT": cin,
        "KERNEL_SIZE": k,
        "STRIDES": strides
    }
    input_shape = [cin, hwin, hwin]
    model = DwConvBnRelu(config_in).get_model()
    try:
        real, pred = profile_and_predict(model, input_shape, mark='')
        reals.append(real)
        preds.append(pred)
    except:
        print("wrong!!", config)

rmse, rmspe, error, acc5, acc10, acc15 = latency_metrics(preds, reals)
for item in zip(preds, reals):
    print(item)
print(f"[Dwconv-bn-relu] rmse: {rmse}, rmspe: {rmspe}, error: {error}, acc5: {acc5}, acc10: {acc10}, acc15: {acc15}")



# se
from modules.blocks.torch.mobilenetv3_block import SE

reals, preds = [], []
configs = [
    [28, 72],
	[28, 160],
	[14, 320], 
	[14, 672], 
	[14, 448], 
	[14, 336], 
	[7, 672],
	[7, 640],
	[7, 480]
]
for i, config in enumerate(configs):
    hwin, cin = config
    config_in = {
        "HW": hwin,
        "CIN": cin
    }
    input_shape = [cin, hwin, hwin]
    model = SE(cin)
    try:
        real, pred = profile_and_predict(model, input_shape, mark='')
        reals.append(real)
        preds.append(pred)
    except:
        print("wrong!!", config)
        
rmse, rmspe, error, acc5, acc10, acc15 = latency_metrics(preds, reals)
for item in zip(preds, reals):
    print(item)
print(f"[SE] rmse: {rmse}, rmspe: {rmspe}, error: {error}, acc5: {acc5}, acc10: {acc10}, acc15: {acc15}")


# class HSwish(nn.Module):
    
#     def __init__(self):
#         super().__init__()
#         self.relu6 = nn.ReLU(6)

#     def forward(self, x):
#         return x * self.relu6(x + 3.) * (1. / 6.)


# model = HSwish()

# pred_lat = predictor.predict(model, "torch", input_shape=(1, 3, 224, 224), apply_nni=False) # in unit of ms

# from modules.blocks.torch.mobilenetv3_block import SE
# model = SE(64)
# profile_and_predict(model, [64, 112, 112], mark="")

