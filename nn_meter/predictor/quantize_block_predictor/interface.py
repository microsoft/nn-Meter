# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from .get_block_arch import get_block_arch_by_name
from .. import load_latency_predictor

class BlockLatencyPredictor:
    def __init__(self, predictor_name):
        self.predictor_name = predictor_name
        if self.predictor_name == "onnx_lut":
            import os, json
            base_dir = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(base_dir, "onnx_lut.json"), 'r') as fp:
                self.predictor = json.load(fp)
        else:
            self.predictor = load_latency_predictor(predictor_name)


    def get_type(self, name, cin, cout, stride, activation):
        '''
        [search candidate block]:
        # MobileNetV1Block
        # MobileNetV2Block_[res/nores]_[relu/hswish]
        # MobileNetV3Block_[res/nores]_[relu/hswish]
        # MobileNetV1DualBlock_[ds/nods]
        # MobileNetV2ResBlock_[res/forceres]_[relu/hswish] always without se
        # MobileNetV3ResBlock_[res/forceres]_[relu/hswish] always with se
        # ResNetBlock_[ds/nods]_[relu/hswish]
        # ResNetSEBlock_[ds/nods]_[relu/hswish]
        # ResNetBugBlock_[ds/nods]

        [simple block]
        # FirstConvBlock_[relu/hswish]
        # FinalExpandBlock_[relu/hswish]
        # FeatureMixBlock_[relu/hswish]
        # LogitsBlock
        '''
        if activation == 'relu6':
            activation = 'relu'
        if name == "MobileNetV1Block" or name == "LogitsBlock":
            return name
        if name == "MobileNetV1DualBlock":
            if cin != cout or stride == 2: return f'{name}_ds'
            else: return f'{name}_nods'
        elif name.startswith("MobileNet"):
            type_list = [name]
            use_res_connect = stride == 1 and cin == cout
            if "ResBlock" in name:
                type_list.append("res" if use_res_connect else "forceres")
            else:
                type_list.append("res" if use_res_connect else "nores")
            type_list.append(activation)
            return "_".join(type_list)
        elif name == "ResNetBlock" or name == "ResNetSEBlock" or name == "ResNetBugBlock":
            type_list = [name]
            use_downsample = stride > 1 or cin != cout
            type_list.append("ds" if use_downsample else "nods" )
            type_list.append(activation)
            return "_".join(type_list)
        else: # FirstConvBlock, FinalExpandBlock, FeatureMixBlock
            return f'{name}_{activation}'


    def get_latency(self, name, hw, cin, cout, kernel_size, expand_ratio, 
                    stride, activation):
        '''
        name:
        hw (int)
        cin (int)
        cout (int)
        kernel_size (int)
        expand_ratio (float)
        stride (int)
        activation: choose from ["relu", "hswish"]
        '''
        if self.predictor_name == "onnx_lut":
            return self.get_latency_by_lut(name, hw, cin, cout, kernel_size, expand_ratio, stride, activation)
        else:
            return self.get_latency_by_predictor(name, hw, cin, cout, kernel_size, expand_ratio, stride, activation)


    def get_latency_by_lut(self, name, hw, cin, cout, kernel_size, expand_ratio, 
                    stride, activation):
        key = f"{name}_{hw}_{cin}_{cout}_{kernel_size}_{expand_ratio}_{stride}_{activation}"
        return self.predictor[key]


    def get_latency_by_predictor(self, name, hw, cin, cout, kernel_size, expand_ratio, 
                    stride, activation):
        type = self.get_type(name, cin, cout, stride, activation)

        # there is only one configs for LogitsBlock, and the predictor has low accuracy for this block    
        if self.predictor_name == "onnxruntime_int8" and type == 'LogitsBlock':
            return 0.16795800000000005

        from nn_meter.predictor.prediction.utils import get_kernel_name
        dicts = get_block_arch_by_name(type, hw, cin, cout, kernel_size, expand_ratio, stride)
        py = 0
        for kernel in dicts:
            kernelname = get_kernel_name(kernel)
            if kernelname in self.predictor.kernel_predictors:
                pred = self.predictor.kernel_predictors[kernelname]
                pys = pred.predict(dicts[kernel]) # in unit of ms
                if len(pys) != 0:
                    py += sum(pys)
        return py
