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
from nn_meter.builder.nn_generator.torch_networks.utils import get_inputs_by_shapes
from nn_meter.builder.kernel_predictor_builder.predictor_builder.utils import get_flops_params

from nas_models.networks.torch.mobilenetv3 import MobileNetV3Net
from nas_models.blocks.torch.mobilenetv3_block import SE


output_path = "/data/jiahang/working/nn-Meter/examples/test_quantize_latency_predictor"
output_name = os.path.join(output_path, "MobilenetV3_test.onnx")

workspace = "/sdc/jiahang/working/ort_mobilenetv3_workspace"
builder_config.init(workspace)
backend = connect_backend(backend_name='ort_cpu_int8')


def profile_model(model, input_shape):
    # print("\n")
    # print(model)
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
    res = backend.profile_model_file(output_name, output_path, input_shape=[[*input_shape]])
 
    return res["latency"].avg

def get_feature(kernel_type, config_dict):
    needed_config = {
        "conv-bn-relu": ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
        "dwconv-bn-relu": ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
        "se": ["HW", "CIN"],
        "hswish": ["HW", "CIN"],
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
from nn_meter.builder.nn_generator.torch_networks.blocks import ConvBnRelu, DwConvBnRelu, HswishBlock, SEBlock

def op_level_test_conv(predictor_name):
    # conv-bn-relu
    with open(predictor_name, "rb") as f:
        predictor = pickle.load(f)

    reals, preds = [], []
    configs = [
        [224, 3, 16, 3, 2], [56, 48, 24, 1, 1], [56, 24, 144, 1, 1], [56, 144, 24, 1, 1], [56, 24, 96, 1, 1], [56, 96, 24, 1, 1],
        [28, 144, 40, 1, 1], [28, 40, 240, 1, 1], [28, 240, 40, 1, 1], [28, 40, 160, 1, 1], [28, 160, 40, 1, 1], [28, 40, 120, 1, 1],
        [28, 120, 40, 1, 1], [14, 160, 80, 1, 1], [14, 80, 320, 1, 1], [14, 320, 80, 1, 1], [14, 80, 480, 1, 1], [14, 480, 112, 1, 1],
        [14, 112, 672, 1, 1], [14, 672, 112, 1, 1], [14, 112, 448, 1, 1], [7, 448, 160, 1, 1], [7, 160, 640, 1, 1], [7, 640, 160, 1, 1],
        [7, 160, 960, 1, 1], [1, 960, 1280, 1, 1], [28, 96, 40, 1, 1], [14, 480, 80, 1, 1], [14, 80, 240, 1, 1], [14, 240, 112, 1, 1],
        [14, 448, 112, 1, 1], [7, 160, 480, 1, 1], [7, 480, 160, 1, 1], [112, 16, 96, 1, 1], [56, 24, 72, 1, 1], [28, 72, 40, 1, 1], 
        [14, 240, 80, 1, 1], [7, 672, 160, 1, 1], [7, 960, 160, 1, 1], [112, 16, 64, 1, 1], [56, 64, 24, 1, 1], [56, 72, 24, 1, 1], 
        [14, 120, 80, 1, 1], [14, 320, 112, 1, 1], [14, 112, 336, 1, 1], [14, 336, 112, 1, 1], [7, 336, 160, 1, 1]
    ]
    for i, config in enumerate(configs):
        hwin, cin, cout, k, strides = config
        config_in = {
            "HW": hwin,
            "CIN": cin,
            "COUT": cout,
            "KERNEL_SIZE": k,
            "STRIDES": strides
        }
        input_shape = [cin, hwin, hwin]
        model = ConvBnRelu(config_in).get_model()
        real = profile_model(model, input_shape)
        pred = predictor.predict([get_feature("conv-bn-relu", config_in)])[0]
        reals.append(real)
        preds.append(pred)

    rmse, rmspe, error, acc10, acc15, acc20 = latency_metrics(preds, reals)
    for item in zip(reals, preds):
        print(item)
    print(f"[Conv-bn-relu] rmse: {rmse}, rmspe: {rmspe}, error: {error}, acc10: {acc10}, acc15: {acc15}, acc20: {acc20}")


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
            
    rmse, rmspe, error, acc10, acc15, acc20 = latency_metrics(preds, reals)
    for item in zip(reals, preds):
        print(item)
    print(f"[Hswish] rmse: {rmse}, rmspe: {rmspe}, error: {error}, acc10: {acc10}, acc15: {acc15}, acc20: {acc20}")


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
    # for i, config in enumerate(configs):
    for i, cin in enumerate(range(630, 650)):
        # hwin, cin, k, strides = config
        hwin, cin, k, strides = 7, cin, 7, 1
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
            
    rmse, rmspe, error, acc10, acc15, acc20 = latency_metrics(preds, reals)
    for cin, res in zip(range(630, 650), reals):
        print(f"cin: {cin}; profiled results: {res}")
    print(f"[Dwconv-bn-relu] rmse: {rmse}, rmspe: {rmspe}, error: {error}, acc10: {acc10}, acc15: {acc15}, acc20: {acc20}")


def op_level_test_se(predictor_name):
    with open(predictor_name, "rb") as f:
        predictor = pickle.load(f)
    # se
    from nas_models.blocks.torch.mobilenetv3_block import SE
    
    reals, preds = [], []
    configs = [
        [28, 72], [28, 160], [14, 320], [14, 672], [14, 448], [14, 336], 
        [7, 672], [7, 640], [7, 480], [112, 16], [28, 120], [14, 120], [14, 480], [14, 480], [14, 240], [14, 240], [14, 320],
        [14, 320], [14, 672], [14, 672], [14, 448], [14, 448], [14, 336], [14, 336], [14, 672],
        [7, 672], [7, 640], [7, 640], [7, 480], [7, 480], [7, 960]
    ]
    for i, config in enumerate(configs):
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
            
    rmse, rmspe, error, acc10, acc15, acc20 = latency_metrics(preds, reals)
    for item in zip(reals, preds):
        print(item)
    print(f"[SE] rmse: {rmse}, rmspe: {rmspe}, error: {error}, acc10: {acc10}, acc15: {acc15}, acc20: {acc20}")

if __name__ == '__main__':
    
    # op_level_test_conv("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/conv-bn-relu_origin.pkl")
    # op_level_test_conv("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/conv-bn-relu_prior2.pkl")
    # op_level_test_conv("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/conv-bn-relu_ofa.pkl")
    # op_level_test_conv("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/conv-bn-relu_ofa_only.pkl")

    # op_level_test_dwconv("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/dwconv-bn-relu_origin.pkl")
    op_level_test_dwconv("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/dwconv-bn-relu_ofa.pkl")
    # op_level_test_dwconv("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/dwconv-bn-relu_ofa_only.pkl")
    
    # op_level_test_hswish("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/hswish_original.pkl")
    # op_level_test_hswish("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/hswish_ofa.pkl")
    # op_level_test_hswish("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/hswish_ofa_only.pkl")

    # op_level_test_se("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/se_original.pkl")
    # op_level_test_se("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/se_ofa.pkl")
    # op_level_test_se("/sdc/jiahang/working/ort_int8_workspace/predictor_build/results/predictors/se_ofa_only.pkl")
    