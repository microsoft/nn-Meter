# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os, json
from collections import defaultdict
from .. import load_latency_predictor

class BlockLatencyPredictor:
    def __init__(self, predictor_name = "mobile_lut"):
        self.predictor_name = predictor_name
        base_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base_dir, "mobile_lut.json"), 'r') as fp:
            self.predictor = json.load(fp)

    def get_latency(self, block_config):
        py = 0
        act = "hard_swish"
        for key, config in block_config.items():
            name = key.split("_")[-1]
            hwin = config['input_feature_map_size']
            hwout = config['feature_map_size']
            ds = config['downsampling']
            cin = config['input_channel']
            cout = config['output_channel']
            if name == "conv":
                for i in range(config["depth"]):
                    s = 2 if ds and i == 0 else 1
                    hw = hwin if i == 0 else hwout
                    exp = config['conv_ratio'][i]
                    ks = config['kernel_size'][i]
                    py += self.predictor[f"{name}_{hw}_{cin}_{cout}_{exp}_{s}_{act}_{ks}"]
            elif name == "transformer":
                for i in range():
                    s = 2 if ds and i == 0 else 1
                    hw = hwin if i == 0 else hwout
                    exp = config['mlp_ratio'][i]
                    v = config['v_scale'][i]
                    py += self.predictor[f"{name}_{hw}_{cin}_{cout}_{exp}_{s}_{act}_{v}"]
        return py


    def get_single_block_arch(self, name, hw, cin, cout, kernel_size, expand_ratio, 
                    stride, activation):
        type = self.get_type(name, cin, cout, stride, activation)
        dicts = get_block_arch_by_name(type, hw, cin, cout, kernel_size, expand_ratio, stride)
        return dicts


    def get_latency_by_predictor(self, block_list):
        raise NotImplementedError # does not support latency predictor now
        from nn_meter.predictor.prediction.utils import get_kernel_name

        # merge dicts
        ops_config = {k: [] for k in self.ops}
        for args in block_list:
            single_block = self.get_single_block_arch(**args)
            for k, v in single_block.items():
                ops_config[k].extend(v)

        py = 0
        for kernel in ops_config:
            if ops_config[kernel] == []:
                continue
            kernelname = get_kernel_name(kernel)
            if kernelname in self.predictor.kernel_predictors:
                pred = self.predictor.kernel_predictors[kernelname]
                pys = pred.predict(ops_config[kernel]) # in unit of ms
                if len(pys) != 0:
                    py += sum(pys)
        return py
