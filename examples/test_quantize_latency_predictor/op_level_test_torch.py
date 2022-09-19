import os
import json
import time
import torch
import pickle
from torch import nn
from nn_meter.predictor.prediction.utils import latency_metrics_acc20 as latency_metrics
from nn_meter.builder.backends import connect_backend
from nn_meter.predictor import load_latency_predictor
from nn_meter.builder import builder_config
from nn_meter.builder.nn_modules.torch_networks.utils import get_inputs_by_shapes
from nn_meter.builder.kernel_predictor_builder.predictor_builder.utils import get_flops_params

from nas_models.networks.torch.mobilenetv3 import MobileNetV3Net
# from nas_models.blocks.torch.mobilenetv3_block import SE
from nas_models.common import make_divisible
# from op_code_torch import SE_xudong, Hswish

output_path = "/data/jiahang/working/nn-Meter/examples/test_quantize_latency_predictor"
output_name = os.path.join(output_path, "test.onnx")

workspace = "/sdc/jiahang/working/ort_mobilenetv3_workspace"
builder_config.init(workspace)


def profile_model(model, input_shape, backend=None):
    # print("\n")
    # print(model)
    # print(model(get_inputs_by_shapes([[*input_shape]], 1)).shape)
    torch.onnx.export(
            model,
            get_inputs_by_shapes([[*input_shape]], 1),
            output_name,
            input_names=['input'],
            output_names=['output'],
            verbose=False,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
        )
    # exit()
    res = backend.profile_model_file(output_name, output_path, input_shape=[[*input_shape]])
 
    return res["latency"].avg

def get_feature(kernel_type, config_dict):
    needed_config = {
        "conv-bn-relu": ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
        "dwconv-bn-relu": ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
        "se": ["HW", "CIN"],
        "hswish": ["HW", "CIN"],
        "avgpool": ["HW", "CIN", "COUT", "KERNEL_SIZE", "POOL_STRIDES"]
    }
    if "COUT" not in config_dict and "COUT" in needed_config[kernel_type]:
        config_dict["COUT"] = config_dict["CIN"]
    feature = [config_dict[data] for data in needed_config[kernel_type]]
    if kernel_type in ["conv-bn-relu", "dwconv-bn-relu"]:
        flop, param = get_flops_params(kernel_type, config_dict)
        flop /= 2e6
        param /= 1e6
        feature.extend([flop, param])
    return feature

## ------------- op level
from nn_meter.builder.nn_modules.torch_networks.blocks import ConvBnRelu, DwConvBnRelu, HswishBlock, SEBlock

def op_level_test_conv(predictor_name):
    # conv-bn-relu
    with open(predictor_name, "rb") as f:
        predictor = pickle.load(f)

    reals, preds = [], []
    configs = [
        # # mobilenet v3
        # [224, 3, 16, 3, 2], [56, 48, 24, 1, 1], [56, 24, 144, 1, 1], [56, 144, 24, 1, 1], [56, 24, 96, 1, 1], [56, 96, 24, 1, 1],
        # [28, 144, 40, 1, 1], [28, 40, 240, 1, 1], [28, 240, 40, 1, 1], [28, 40, 160, 1, 1], [28, 160, 40, 1, 1], [28, 40, 120, 1, 1],
        # [28, 120, 40, 1, 1], [14, 160, 80, 1, 1], [14, 80, 320, 1, 1], [14, 320, 80, 1, 1], [14, 80, 480, 1, 1], [14, 480, 112, 1, 1],
        # [14, 112, 672, 1, 1], [14, 672, 112, 1, 1], [14, 112, 448, 1, 1], [7, 448, 160, 1, 1], [7, 160, 640, 1, 1], [7, 640, 160, 1, 1],
        # [7, 160, 960, 1, 1], [1, 960, 1280, 1, 1], [28, 96, 40, 1, 1], [14, 480, 80, 1, 1], [14, 80, 240, 1, 1], [14, 240, 112, 1, 1],
        # [14, 448, 112, 1, 1], [7, 160, 480, 1, 1], [7, 480, 160, 1, 1], [112, 16, 96, 1, 1], [56, 24, 72, 1, 1], [28, 72, 40, 1, 1], 
        # [14, 240, 80, 1, 1], [7, 672, 160, 1, 1], [7, 960, 160, 1, 1], [112, 16, 64, 1, 1], [56, 64, 24, 1, 1], [56, 72, 24, 1, 1], 
        # [14, 120, 80, 1, 1], [14, 320, 112, 1, 1], [14, 112, 336, 1, 1], [14, 336, 112, 1, 1], [7, 336, 160, 1, 1]

        # # conv1x1, expand ratio 3 4 5 6 
        # [56, 16, 48, 1, 1], [56, 32, 96, 1, 1], 
        # [56, 40, 120, 1, 1], [56, 48, 144, 1, 1], 
        # [56, 56, 168, 1, 1], [56, 64, 192, 1, 1],
        # [56, 96, 288, 1, 1], [56, 128, 384, 1, 1], [56, 160, 480, 1, 1], [56, 240, 720, 1, 1], [56, 320, 960, 1, 1], [56, 480, 1440, 1, 1],
        # [56, 16, 64, 1, 1], [56, 32, 128, 1, 1], 
        # [56, 40, 160, 1, 1], [56, 48, 192, 1, 1], 
        # [56, 56, 224, 1, 1], [56, 64, 256, 1, 1],
        # [56, 96, 384, 1, 1], [56, 128, 512, 1, 1], [56, 160, 640, 1, 1], [56, 240, 960, 1, 1], [56, 320, 1280, 1, 1], [56, 480, 1920, 1, 1],
        # [56, 16, 80, 1, 1], [56, 32, 160, 1, 1], 
        # [56, 40, 200, 1, 1], [56, 48, 240, 1, 1], 
        # [56, 56, 280, 1, 1], [56, 64, 320, 1, 1],
        # [56, 96, 480, 1, 1], [56, 128, 640, 1, 1], [56, 160, 800, 1, 1], [56, 240, 1200, 1, 1], [56, 320, 1600, 1, 1], [56, 480, 2400, 1, 1],
        # [56, 16, 96, 1, 1], [56, 32, 192, 1, 1], 
        # [56, 40, 240, 1, 1], [56, 48, 288, 1, 1], 
        # [56, 56, 336, 1, 1], [56, 64, 384, 1, 1],
        # [56, 96, 576, 1, 1], [56, 128, 768, 1, 1], [56, 160, 960, 1, 1], [56, 240, 1440, 1, 1], [56, 320, 1920, 1, 1], [56, 480, 2880, 1, 1]
        # [56, 560, 1680, 1, 1], [56, 560, 2240, 1, 1], [56, 560, 2800, 1, 1], [56, 560, 3360, 1, 1],
        # [56, 72, 288, 1, 1], [56, 80, 320, 1, 1]
        # [28, 16, 64, 1, 1], [28, 32, 128, 1, 1], [28, 40, 160, 1, 1], [28, 48, 192, 1, 1], 
        [28, 56, 224, 1, 1], [28, 64, 256, 1, 1],
        [28, 72, 288, 1, 1], [28, 80, 320, 1, 1], 
        # [28, 96, 384, 1, 1], [28, 128, 512, 1, 1], [28, 160, 640, 1, 1], [28, 240, 960, 1, 1], 
        # [28, 320, 1280, 1, 1], [28, 480, 1920, 1, 1], [28, 560, 2240, 1, 1]
    ]
    
    # # configs in MobilenetV3Large
    # configs = [
    #     [224, 3, 16, 3, 2], [112, 16, 16, 1, 1], [112, 16, 64, 1, 1], [56, 64, 24, 1, 1], [56, 24, 72, 1, 1], [56, 72, 24, 1, 1], [56, 24, 72, 1, 1],
    #     [28, 72, 40, 1, 1], [28, 40, 120, 1, 1], [28, 120, 40, 1, 1], [28, 40, 120, 1, 1], [28, 120, 40, 1, 1], [28, 40, 240, 1, 1], [14, 240, 80, 1, 1], 
    #     [14, 80, 200, 1, 1], [14, 200, 80, 1, 1], [14, 80, 184, 1, 1], [14, 184, 80, 1, 1], [14, 80, 184, 1, 1], [14, 184, 80, 1, 1], [14, 80, 480, 1, 1],
    #     [14, 480, 112, 1, 1], [14, 112, 672, 1, 1], [14, 672, 112, 1, 1], [14, 112, 672, 1, 1], [7, 672, 160, 1, 1], [7, 160, 960, 1, 1], [7, 960, 160, 1, 1],
    #     [7, 160, 960, 1, 1], [7, 960, 160, 1, 1], [7, 160, 960, 1, 1]
    # ]
    for i, config in enumerate(configs):
    # for i, cout in enumerate(range(600, 681)):
    # for i, ks in enumerate([1, 3, 5, 7]):
    # for i, c in enumerate([16, 32, 48, 64, 96, 128, 160, 240, 320, 480, 560]):
        hwin, cin, cout, k, strides = config
        
        # hwin, cin, cout, k, strides = 28, 640, cout, 3, 1
        
        # hwin, cin, cout, k, strides = 14, 320, 320, ks, 1
        # hwin, cin, cout, k, strides = 56, 32, 32, ks, 1
        # hwin, cin, cout, k, strides = 56, 96, 96, ks, 1
        
        # hwin, cin, cout, k, strides = 56, c, c, 1, 1
        
        config_in = {
            "HW": hwin,
            "CIN": cin,
            "COUT": cout,
            "KERNEL_SIZE": k,
            "STRIDES": strides
        }
        # print(config_in)
        input_shape = [cin, hwin, hwin]
        model = ConvBnRelu(config_in).get_model()
        real = profile_model(model, input_shape, backend)
        pred = predictor.predict([get_feature("conv-bn-relu", config_in)])[0]
        reals.append(real)
        preds.append(pred)
        # print(real, pred)
        print(real)
        # break

    rmse, rmspe, error, acc10, acc15, acc20 = latency_metrics(preds, reals)
    
    # print(backend_name, "Conv")
    # for item in zip(reals, preds):
    #     print(item[0])
    
    # for cin, res in zip(range(600, 681), reals):
    #     print(f"{cin}; {res}")
        
    # for ks, res in zip([1, 3, 5, 7], reals):
    #     print(f"{ks}; {res}")
    
    # for c, res in zip([16, 32, 48, 64, 96, 128, 160, 240, 320, 480, 560], reals):
    #     print(f"{c}, {res}")
    
    print(f"[Conv-bn-relu] rmse: {rmse}, rmspe: {rmspe}, error: {error}, acc10: {acc10}, acc15: {acc15}, acc20: {acc20}")


def op_level_test_conv_group():
    for i, ks in enumerate([1, 3, 5, 7]):
        for group in [1, 16, 32, 64, 128, 640]:
            # hwin, cin, cout, k, strides = 56, 96, 96, ks, 1
            hwin, cin, cout, k, strides = 14, 640, 640, ks, 1
            
            config_in = {
                "HW": hwin,
                "CIN": cin,
                "COUT": cout,
                "KERNEL_SIZE": k,
                "STRIDES": strides,
                "GROUPS": group
            }
            # print(config_in)
            input_shape = [cin, hwin, hwin]
            model = ConvBnRelu(config_in).get_model()
            real = profile_model(model, input_shape)
            print(f'{ks}, {group}, {real}')


def op_level_test_hswish(predictor_name):
    with open(predictor_name, "rb") as f:
        predictor = pickle.load(f)
    # # hswish
    # class HSwishForParse(nn.Module):
        
    #     def __init__(self):
    #         super().__init__()
    #         self.relu6 = nn.ReLU(6)

    #     def forward(self, x):
    #         return x * self.relu6(x + 3.) * (1. / 6.)

    reals, preds = [], []
    configs = [
        [112, 16], [28, 120], [14, 120], [14, 480], [14, 480], [14, 240], [14, 240], [14, 320],
        [14, 320], [14, 672], [14, 672], [14, 448], [14, 448], [14, 336], [14, 336], [14, 672],
        [7, 672], [7, 640], [7, 640], [7, 480], [7, 480], [7, 960], [1, 1280],
    ]

    # # configs in MobilenetV3Large
    # configs = [
    #     [112, 16], [28, 240], [14, 240], [14, 200], [14, 200], [14, 184], [14, 184], [14, 184],
    #     [14, 184], [14, 480], [14, 480], [14, 672], [14, 672], [14, 672], [7, 672], [7, 960], [7, 960],
    #     [7, 960], [7, 960], [7, 960], [1, 1280]
    # ]
    for i, config in enumerate(configs):
        hwin, cin = config
        config_in = {
            "HW": hwin,
            "CIN": cin
        }
        input_shape = [cin, hwin, hwin]
        model = HswishBlock(config_in).get_model()
        real = profile_model(model, input_shape)
        pred = predictor.predict([get_feature("hswish", config_in)])[0]
        reals.append(real)
        preds.append(pred)
        print(real, pred)
        # break
            
    rmse, rmspe, error, acc10, acc15, acc20 = latency_metrics(preds, reals)
    # print(backend_name, "Hswish")
    # for item in zip(reals, preds):
    #     print(item[0])
    print(f"[Hswish] rmse: {rmse}, rmspe: {rmspe}, error: {error}, acc10: {acc10}, acc15: {acc15}, acc20: {acc20}")


def op_level_test_swish():
    # # swish
    class SwishForParse(nn.Module):
        
        def __init__(self):
            super().__init__()
            # self.swish = nn.SiLU()

        def forward(self, x):
            return x * torch.sigmoid(x)
            # return self.swish(x)

    # configs in MobilenetV3Large
    # configs = [
    #     [112, 16], [28, 240], [14, 240], [14, 200], [14, 200], [14, 184], [14, 184], [14, 184],
    #     [14, 184], [14, 480], [14, 480], [14, 672], [14, 672], [14, 672], [7, 672], [7, 960], [7, 960],
    #     [7, 960], [7, 960], [1, 1280]
    # ]
    configs = [[112, 16]
            #    , [56, 40], [28, 80], [14, 160], [7, 320]
               ]
    for i, config in enumerate(configs):
        hwin, cin = config
        config_in = {
            "HW": hwin,
            "CIN": cin
        }
        input_shape = [cin, hwin, hwin]
        model = SwishForParse()
        real = profile_model(model, input_shape, backend=backend)
        print(real)

def op_level_test_dwconv(predictor_name):
    with open(predictor_name, "rb") as f:
        predictor = pickle.load(f)
    # dwconv-bn-relu
    reals, preds = [], []
    configs = [
        [112, 16, 3, 1], [112, 48, 3, 2], [56, 144, 3, 1], [56, 96, 5, 1], [56, 144, 5, 2], [28, 240, 3, 1], [28, 160, 7, 1],
        [28, 120, 3, 1], [28, 160, 3, 2], [14, 320, 5, 1], [14, 480, 3, 1], [14, 672, 3, 1], [14, 448, 3, 2], [7, 640, 7, 1],
        [7, 640, 3, 1], [7, 640, 5, 1], [56, 96, 7, 2], [28, 240, 7, 1], [28, 160, 5, 2], [14, 240, 5, 1], [14, 448, 7, 1],
        [14, 448, 7, 2], [7, 480, 5, 1], [112, 96, 3, 2], [56, 144, 5, 1], [56, 72, 3, 2], [28, 240, 5, 1], [28, 160, 5, 1],
        [28, 240, 7, 2], [14, 480, 7, 1], [14, 320, 7, 1], [7, 480, 7, 1], [28, 120, 7, 1], [14, 240, 7, 1], [14, 448, 5, 1],
        [14, 672, 3, 2], [7, 960, 5, 1], [7, 480, 3, 1], [112, 64, 3, 2], [56, 72, 5, 1], [56, 144, 7, 1], [56, 96, 3, 1],
        [56, 144, 3, 2], [28, 120, 5, 2], [14, 320, 3, 1], [14, 448, 3, 1], [14, 672, 7, 2], [7, 960, 3, 1], [56, 96, 7, 1],
        [56, 72, 7, 1], [56, 72, 7, 2], [28, 120, 5, 1], [28, 160, 7, 2], [14, 672, 5, 1], [14, 672, 5, 2], [7, 960, 7, 1],
        [28, 120, 7, 2], [14, 240, 3, 1], [14, 480, 5, 1], [14, 336, 5, 1], [112, 48, 5, 2], [28, 160, 3, 1], [14, 336, 7, 2],
        [56, 72, 3, 1], [56, 72, 5, 2], [28, 240, 3, 2], [14, 336, 7, 1], [56, 96, 3, 2], [56, 96, 5, 2], [14, 336, 5, 2],
        [56, 144, 7, 2], [112, 96, 5, 2], [14, 448, 5, 2], [14, 336, 3, 1], [112, 64, 5, 2], [28, 240, 5, 2], [14, 336, 3, 2],
        [28, 120, 3, 2], [112, 48, 7, 2], [14, 672, 7, 1], [112, 64, 7, 2], [112, 96, 7, 2]
    ]
    # # configs in MobilenetV3Large
    # configs = [
    #     [112, 16, 16, 3, 1], [112, 64, 64, 3, 2], [56, 72, 72, 3, 1], [56, 72, 72, 5, 2], [28, 120, 120, 5, 1], [28, 120, 120, 5, 1],
    #     [28, 240, 240, 3, 2], [14, 200, 200, 3, 1], [14, 184, 184, 3, 1], [14, 184, 184, 3, 1], [14, 480, 480, 3, 1], [14, 672, 672, 3, 1],
    #     [14, 672, 672, 5, 2], [7, 960, 960, 5, 1], [7, 960, 960, 5, 1]
    # ]
    for i, config in enumerate(configs):
    # for i, cin in enumerate(range(600, 681)):
    # for i, ks in enumerate([1, 3, 5, 7]):
        hwin, cin, k, strides = config
        # hwin, cin, k, strides = 28, cin, 3, 1
        # hwin, cin, k, strides = 14, 320, ks, 1
        # hwin, cin, k, strides = 56, 32, ks, 1
        # hwin, cin, k, strides = 56, 96, ks, 1
        config_in = {
            "HW": hwin,
            "CIN": cin,
            "COUT": cin,
            "KERNEL_SIZE": k,
            "STRIDES": strides
        }
        input_shape = [cin, hwin, hwin]
        model = DwConvBnRelu(config_in).get_model()
        real = profile_model(model, input_shape)
        pred = predictor.predict([get_feature("dwconv-bn-relu", config_in)])[0]
        reals.append(real)
        preds.append(pred)
        print(real, pred)
        # break
            
    rmse, rmspe, error, acc10, acc15, acc20 = latency_metrics(preds, reals)
    # print(backend_name, "Dwconv")
    # for cin, res in zip(range(600, 681), reals):
    #     print(f"{cin}; {res}")
    # for item in zip(reals, preds):
    #     print(item[0])
    # for ks, res in zip([1, 3, 5, 7], reals):
    #     print(f"{ks}; {res}")
    print(f"[Dwconv-bn-relu] rmse: {rmse}, rmspe: {rmspe}, error: {error}, acc10: {acc10}, acc15: {acc15}, acc20: {acc20}")



def op_level_test_se(predictor_name):
    # from op_code_torch import SE_xudong
    from nn_meter.builder.nn_generator.torch_networks.blocks import SEBlock
    with open(predictor_name, "rb") as f:
        predictor = pickle.load(f)

    reals, preds = [], []
    configs = [
        [28, 72], [28, 160], [14, 320], [14, 672], [14, 448], [14, 336], 
        [7, 672], [7, 640], [7, 480], [112, 16], [28, 120], [14, 120], [14, 480], [14, 480], [14, 240], [14, 240], [14, 320],
        [14, 320], [14, 672], [14, 672], [14, 448], [14, 448], [14, 336], [14, 336], [14, 672],
        [7, 672], [7, 640], [7, 640], [7, 480], [7, 480], [7, 960]
    ]
    # # configs in MobilenetV3Large
    # configs = [
    #     [28, 72], [28, 120], [28, 120], [14, 480], [14, 672], [7, 672], [7, 960], [7, 960]
    # ]
    # for i, cin in enumerate(range(600, 681)):
    for i, config in enumerate(configs):
        # hwin, cin = 14, cin
        hwin, cin = config
        config_in = {
            "HW": hwin,
            "CIN": cin
        }
        input_shape = [cin, hwin, hwin]
        model = SEBlock(config_in).get_model()
        real = profile_model(model, input_shape)
        pred = predictor.predict([get_feature("se", config_in)])[0]
        reals.append(real)
        preds.append(pred)
        # break
            
    rmse, rmspe, error, acc10, acc15, acc20 = latency_metrics(preds, reals)
    print(backend_name, "SE")
    for item in zip(reals, preds):
        print(item)
    # for cin, res in zip(range(600, 681), reals):
    #     print(f"{cin}; {res}")
    print(f"[SE] rmse: {rmse}, rmspe: {rmspe}, error: {error}, acc10: {acc10}, acc15: {acc15}, acc20: {acc20}")


def op_level_test_avgpool(predictor_name):
    with open(predictor_name, "rb") as f:
        predictor = pickle.load(f)
    
    reals, preds = [], []
    configs = [
        [56, 64, 1, 1], [56, 256, 2, 2], [28, 512, 2, 2], [14, 816, 2, 2]
    ]
    for i, config in enumerate(configs):
        hwin, cin, ks, s = config
        config_in = {
            "HW": hwin,
            "CIN": cin,
            "COUT": cin,
            "KERNEL_SIZE": ks,
            "POOL_STRIDES": s
        }
        input_shape = [cin, hwin, hwin]
        model = nn.AvgPool2d(kernel_size=ks, stride=s, padding=0, ceil_mode=True)
        real = profile_model(model, input_shape)
        pred = predictor.predict([get_feature("avgpool", config_in)])[0]
        reals.append(real)
        preds.append(pred)
        # break
            
    rmse, rmspe, error, acc10, acc15, acc20 = latency_metrics(preds, reals)
    for item in zip(reals, preds):
        print(item)
    print(f"[Avgpool] rmse: {rmse}, rmspe: {rmspe}, error: {error}, acc10: {acc10}, acc15: {acc15}, acc20: {acc20}")


def op_level_test_mobilenetv3_large():
    # from torchvision.models import mobilenet_v3_large
    from torchvision_mobilenetv3 import mobilenet_v3_large
    # from tensorflow.keras.applications import MobileNetV3Large
    model = mobilenet_v3_large()
    input_shape = [3, 224, 224]
    # real = profile_model(model, input_shape)
    # model = SE_xudong(64)
    # model = Hswish()
    
    predictor_name = "onnxruntime_int8"
    predictor = load_latency_predictor(predictor_name)
    pred_lat = predictor.predict(model, "torch", input_shape=(1, 3, 224, 224), apply_nni=False)
    # pred_lat = predictor.predict(model, "torch", input_shape=(1, 64, 56, 56), apply_nni=False)
    
    # print(real, pred_lat)

    # model = tf.keras.applications.MobilenetV3Large()
    

def op_level_test_cascade_mbv1():
    from op_code_torch import res_block, seq_block
    i = 3
    configs = [
        # 112x112x16->56x56x32
        ['112x112x16->56x56x32', 'sequential', 'ks{i}', [112, 16, 32, i, 1, 32, 32, i, 2]],
        ['112x112x16->56x56x32', 'sequential', 'ks{i}', [112, 16, 16, i, 1, 16, 32, i, 2]],
        ['112x112x16->56x56x32', 'sequential', 'ks{i}', [112, 16, 32, i, 2, 32, 32, i, 1]],
        ['112x112x16->56x56x32', 'sequential', 'ks{i}', [112, 16, 16, i, 2, 16, 32, i, 1]],

        # 56x56x32->56x56x32
        ['56x56x32->56x56x32', 'res_connected', 'ks{i}', [56, 32, 32, i, 1]],
        ['56x56x32->56x56x32', 'sequential', 'ks{i}', [56, 32, 32, i, 1, 32, 32, i, 1]],
        
        # 56x56x32->28x28x64
        ['56x56x32->28x28x64', 'sequential', 'ks{i}', [56, 32, 64, i, 1, 64, 64, i, 2]],
        ['56x56x32->28x28x64', 'sequential', 'ks{i}', [56, 32, 32, i, 1, 32, 64, i, 2]],
        ['56x56x32->28x28x64', 'sequential', 'ks{i}', [56, 32, 64, i, 2, 64, 64, i, 1]],
        ['56x56x32->28x28x64', 'sequential', 'ks{i}', [56, 32, 32, i, 2, 32, 64, i, 1]],

        # 28x28x64->28x28x64
        ['28x28x64->28x28x64', 'res_connected', 'ks{i}', [28, 64, 64, i, 1]],
        ['28x28x64->28x28x64', 'sequential', 'ks{i}', [28, 64, 64, i, 1, 64, 64, i, 1]],
        
        # 28x28x64->14x14x128
        ['28x28x64->14x14x128', 'sequential', 'ks{i}', [28, 64, 64, i, 1, 64, 128, i, 2]],
        ['28x28x64->14x14x128', 'sequential', 'ks{i}', [28, 64, 128, i, 1, 128, 128, i, 2]],
        ['28x28x64->14x14x128', 'sequential', 'ks{i}', [28, 64, 64, i, 2, 64, 128, i, 1]],
        ['28x28x64->14x14x128', 'sequential', 'ks{i}', [28, 64, 128, i, 2, 128, 128, i, 1]],

        # 14x14x128->14x14x128
        ['14x14x128->14x14x128', 'res_connected', 'ks{i}', [14, 128, 128, i, 1]],
        ['14x14x128->14x14x128', 'sequential', 'ks{i}', [14, 128, 128, i, 1, 128, 128, i, 1]],
        
        # 14x14x128->7x7x256
        ['14x14x128->7x7x256', 'sequential', 'ks{i}', [14, 128, 128, i, 1, 128, 256, i, 2]],
        ['14x14x128->7x7x256', 'sequential', 'ks{i}', [14, 128, 256, i, 1, 256, 256, i, 2]],
        ['14x14x128->7x7x256', 'sequential', 'ks{i}', [14, 128, 128, i, 2, 128, 256, i, 1]],
        ['14x14x128->7x7x256', 'sequential', 'ks{i}', [14, 128, 256, i, 2, 256, 256, i, 1]],

        # 7x7x256->7x7x256
        ['7x7x256->7x7x256', 'res_connected', 'ks{i}', [7, 256, 256, i, 1]],
        ['7x7x256->7x7x256', 'sequential', 'ks{i}', [7, 256, 256, i, 1, 256, 256, i, 1]],
        
    ]    
    
    for i, config in enumerate(configs):
        name, block, ks, param = config
        if block == 'res_connected':
            for ks_v in [3, 5]:
                hwin, cin, cout, ks, s = param
                input_shape = [cin, hwin, hwin]
                model = res_block(cin, cout, ks_v, s)
                real = profile_model(model, input_shape)
                print(f'{backend_name}, {name}, {block}, ks{ks_v}, cin_{cin}_cout_{cout}, {real}')
        else:
            for ks_v in [3, 5]:
                hwin, cin1, cout1, ks1, s1, cin2, cout2, ks2, s2 = param
                
                input_shape = [cin1, hwin, hwin]
                model = seq_block(cin1, cout1, ks_v, s1, cin2, cout2, ks_v, s2)
                real = profile_model(model, input_shape)
                print(f'{backend_name}, {name}, {block}, ks{ks_v}, cin1_{cin1}_cout1_{cout1}_s1_{s1}_cin2_{cin2}_cout2_{cout2}_s2_{s2}, {real}')


    
if __name__ == '__main__':
    backend_name = 'ort_cpu'
    backend = connect_backend(backend_name)
    
    op_level_test_conv("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/conv-bn-relu.pkl")
    # op_level_test_conv_group()

    # op_level_test_dwconv("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/dwconv-bn-relu_origin.pkl")
    # op_level_test_dwconv("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/dwconv-bn-relu_ofa.pkl")
    # op_level_test_dwconv("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/dwconv-bn-relu_ofa_only.pkl")
    # op_level_test_dwconv("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/dwconv-bn-relu_finegrained1.pkl")
    # op_level_test_dwconv("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/dwconv-bn-relu_finegrained1_filt8.pkl")
    # op_level_test_dwconv("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/dwconv-bn-relu_refined16.pkl")
    
    # op_level_test_hswish("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/hswish_original.pkl")
    # op_level_test_hswish("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/hswish_ofa.pkl")
    # op_level_test_hswish("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/hswish.pkl")
    # op_level_test_hswish("/sdc/jiahang/working/ort_mobilenetv3_workspace/predictor/hswish.pkl")
    
    # op_level_test_se("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/se_original.pkl")
    # op_level_test_se("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/se_ofa.pkl")
    # op_level_test_se("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/se.pkl")
    # op_level_test_se("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/se.pkl")

    # op_level_test_avgpool("/sdc/jiahang/working/ort_mobilenetv3_workspace/predictor/avgpool.pkl")
    # op_level_test_mobilenetv3_large()
    
    # op_level_test_conv("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/conv-bn-relu_ofa_only.pkl")
    # op_level_test_dwconv("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/dwconv-bn-relu_finegrained1_filt8.pkl")    
    # op_level_test_hswish("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/hswish_ofa_only.pkl")
    # op_level_test_se("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/se_original.pkl")

    # backend_name = 'ort_cpu'
    # backend = connect_backend(backend_name)
    # op_level_test_cascade_mbv1()
    # op_level_test_swish()
    
    # backend_name = 'ort_cpu_int8'
    # backend = connect_backend(backend_name)
    # op_level_test_cascade_mbv1()