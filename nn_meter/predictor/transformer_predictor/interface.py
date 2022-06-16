# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os, json

class BlockLatencyPredictor:
    def __init__(self, predictor_name = "mobile_lut"):
        self.predictor_name = predictor_name
        base_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base_dir, "mobile_lut.json"), 'r') as fp:
            self.predictor = json.load(fp)

    def get_latency(self, block_config):
        '''
        sample = (
            176, # 0 input res
            (0, 0, 0, 1, 1, 1), # 1 block type
            (24, 40, 80, 112, 192, 320), # 2 channels
            (2, 3, 2, 2, 3, 4), # 3 depths
            (6, 6, 6, 6, 6, 4, 4), # 4 conv expansion ratio
            (3, 3, 3, 3, 3, 5, 5), # 5 conv kr size
            (2, 2, 4, 4, 4, 4, 4, 4, 4), # 6 trans mlp ratio
            (14, 14, 24, 24, 24, 40, 40, 40, 40), # 7 trans num heads
            (1, 1, 1, 1, 1, 1, 1, 1, 1), # 8 windows size
            (1, 1, 1, 1, 1, 1, 1, 1, 1), # 9 qk scale
            (2, 2, 2, 2, 2, 2, 2, 2, 2) # 10 v scale
        )
        '''
        py = 0
        act = "hard_swish"
        strides = (2, 2, 2, 1, 2, 2)

        # first_block
        hw = block_config[0]
        py += self.predictor[f"conv_{hw}_3_16_1_2_{act}_3"]
        # print(f"conv_{hw}_3_16_1_2_{act}_3")
        hw = hw // 2
        stage_cout = 16

        # layer_choice blocks
        conv_count, trans_count = 0, 0
        for stage_idx, block_type in enumerate(block_config[1]):
            name = "conv" if block_type == 0 else "transformer"
            stage_stride = strides[stage_idx]
            stage_hwin = hw
            stage_hwout = hw // stage_stride if hw % stage_stride == 0 else hw // stage_stride + 1
            hw = stage_hwout
            stage_cin = stage_cout
            stage_cout = block_config[2][stage_idx]
            if name == "conv":
                for i in range(block_config[3][stage_idx]):
                    s = stage_stride if i == 0 else 1
                    layer_hw = stage_hwin if i == 0 else stage_hwout
                    cin = stage_cin if i == 0 else stage_cout
                    cout = stage_cout
                    exp = block_config[4][conv_count]
                    ks = block_config[5][conv_count]
                    conv_count += 1

                    # predict by lut
                    py += self.predictor[f"{name}_{layer_hw}_{cin}_{cout}_{exp}_{s}_{act}_{ks}"]
                    # print(f"{name}_{layer_hw}_{cin}_{cout}_{exp}_{s}_{act}_{ks}")

            elif name == "transformer":
                for i in range(block_config[3][stage_idx]):
                    s = stage_stride if i == 0 else 1
                    ds = "ds" if i == 0 else "nods"
                    layer_hw = stage_hwin if i == 0 else stage_hwout
                    cin = stage_cin if i == 0 else stage_cout
                    cout = stage_cout
                    exp = block_config[6][trans_count]
                    v = block_config[10][trans_count]
                    trans_count += 1

                    # predict by lut
                    py += self.predictor[f"{name}_{layer_hw}_{cin}_{cout}_{exp}_{s}_{act}_{v}_{ds}"]
                    # print(f"{name}_{layer_hw}_{cin}_{cout}_{exp}_{s}_{act}_{v}_{ds}")

        assert conv_count == len(block_config[5])
        assert trans_count == len(block_config[6])

        return py


    def get_single_block_arch(self, name, hw, cin, cout, kernel_size, expand_ratio, 
                    stride, activation):
        raise NotImplementedError # does not support latency predictor now
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
