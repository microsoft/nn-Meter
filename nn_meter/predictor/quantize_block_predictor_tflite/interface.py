# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from .get_block_arch import get_block_arch_by_name
from .. import load_latency_predictor

class BlockLatencyPredictor:
    def __init__(self, predictor_name):
        self.predictor_name = predictor_name
        self.predictor = load_latency_predictor(predictor_name)
        
    def get_type(self, name, cin, cout, stride, activation, use_se):
        '''
        [mobilenet block] #17:   
        # mobilenetv1
        # mobilenetv2_[res/nores]_[se/nose]_[relu/hswish]
        # mobilenetv3_[res/nores]_[se/nose]_[relu/hswish]

        [resnet block] #4
        # resnet_[ds/nods]_[relu/hswish]
        
        [simple block] #3
        # first_conv_[relu/hswish]
        # logits_block
        '''
        if name == "mobilenetv1":
            return name
        elif name.startswith("mobilenet"):
            type_list = [name]
            use_res_connect = stride == 1 and cin == cout
            type_list.append("res" if use_res_connect else "nores" )
            type_list.append("se" if use_se else "nose" )
            type_list.append(activation)
            return "_".join(type_list)
        elif name == "resnet":
            type_list = ["resnet"]
            use_downsample = stride > 1 or cin != cout
            type_list.append("ds" if use_downsample else "nods" )
            type_list.append(activation)
            return "_".join(type_list)
        elif name == "first_conv":
            return f'{name}_{activation}'
        elif name == "logits_block":
            return name

    def get_latency(self, name, hw, cin, cout, kernel_size, expand_ratio, 
                    stride, activation, use_se):
        '''
        name: choose from ["mobilenetv1", "mobilenetv2", "mobilenetv3", "resnet", "first_conv", "logits_block"]
        hw (int)
        cin (int)
        cout (int)
        kernel_size (int)
        expand_ratio (float)
        stride (int)
        activation: choose from ["relu", "hswish"]
        use_se (Boolean)
        '''
        type = self.get_type(name, cin, cout, stride, activation, use_se)
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
