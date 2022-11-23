# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os, json

class BlockLatencyPredictor:
    def __init__(self, predictor_name = "pixel6_lut", layer_norm = True, mode = "layerwise", silence = True):
        self.predictor_name = predictor_name
        base_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base_dir, "lut", f"{predictor_name}_ln_v2.json"), 'r') as fp:
            self.predictor = json.load(fp)
        self.layer_norm = layer_norm
        self.mode = mode
        self.silence = silence

    def get_latency(self, block_config):
        '''
        arch = (
            224, # 0 input res
            (16, 24, 40, 64, 112, 192, 320), # 1 channels
            (1, 3, 4, 2, 3, 4, 5), # 2 depths
            (1, 5, 5, 5, 6, 6, 6, 6), # 3 conv expansion ratio
            (3, 5, 5, 5, 5, 5, 5, 5), # 4 conv kr size
            (4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3), # 5 trans mlp ratio
            (4, 4, 7, 7, 7, 12, 12, 12, 12, 20, 20, 20, 20, 20), # 6 trans num heads
            (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), # 7 windows size
            (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), # 8 qk scale
            (2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4) # 9 v scale
        )
        '''
        py = 0
        act = "hard_swish"
        strides = (1, 2, 2, 2, 2, 1, 2) # didn't contain the first conv3x3
        use_se = (False, False, True)

        # first_block
        hw = block_config[0]
        key = f"firstconv_{hw}_3_{block_config[1][0]}_2_3"
        py += self.predictor[key]
        if not self.silence: print(key)
        hw = hw // 2
        stage_cout = 16

        # layer_choice blocks
        conv_count, trans_count = 0, 0
        for stage_idx, channel in enumerate(block_config[1]):
            name = "conv" if stage_idx <= 2 else "transformer"
            stage_stride = strides[stage_idx]
            stage_hwin = hw
            stage_hwout = hw // stage_stride
            hw = stage_hwout
            stage_cin = stage_cout
            stage_cout = channel
            if name == "conv":
                for i in range(block_config[2][stage_idx]):
                    s = stage_stride if i == 0 else 1
                    layer_hw = stage_hwin if i == 0 else stage_hwout
                    cin = stage_cin if i == 0 else stage_cout
                    cout = stage_cout
                    exp = block_config[3][conv_count]
                    ks = block_config[4][conv_count]
                    se = use_se[stage_idx]
                    conv_count += 1

                    # predict by lut
                    key = f"{name}_{layer_hw}_{cin}_{cout}_{exp}_{s}_{act}_{ks}_{'se' if se else 'nose'}"
                    py += self.predictor[key]
                    if not self.silence: print(key)

            elif name == "transformer":
                for i in range(block_config[2][stage_idx]):
                    s = stage_stride if i == 0 else 1
                    ds = "ds" if i == 0 else "nods"
                    layer_hw = stage_hwin if i == 0 else stage_hwout
                    cin = stage_cin if i == 0 else stage_cout
                    cout = stage_cout
                    exp = block_config[5][trans_count]
                    v = block_config[9][trans_count] if len(block_config) > 9 else 4
                    trans_count += 1

                    if self.mode == "layerwise": # predict by lut
                        ds_exp_mark = "_6" if i == 0 else ""
                        key = f"{name}_{layer_hw}_{cin}_{cout}_{exp}_{s}_{act}_{v}_{ds}{ds_exp_mark}_{'ln' if self.layer_norm else 'bn'}"
                        lpy = self.predictor[key]
                        if not self.silence: print(key)

                        py += lpy

                    else: # predict by attn/ffn lut
                        tpy = 0

                        # downsample
                        if ds == "ds":
                            key = f"transds_{layer_hw}_{cin}_{cout}_{s}_6"
                            tpy += self.predictor[key]
                            if not self.silence: print(key)
                            layer_hw = stage_hwout

                        # attn
                        key = f'transattn_{layer_hw}_{cout}_{act}_{v}_{"ln" if self.layer_norm else "bn"}'
                        tpy += self.predictor[key]
                        if not self.silence: print(key)

                        # ffn
                        key = f'transffn_{layer_hw}_{cout}_{exp}_{act}_{"ln" if self.layer_norm else "bn"}'
                        tpy += self.predictor[key]
                        if not self.silence: print(key)

                        py += tpy

            layer_hw = stage_hwout

        # MBPool block
        key = f"mbpool_{layer_hw}_{block_config[1][-1]}_1984_6_{act}"
        py += self.predictor[key]
        if not self.silence: print(key)

        assert conv_count == len(block_config[4])
        assert trans_count == len(block_config[5])

        return py


    def get_nasvit_latency(self, block_config):
        '''
        arch = (
            224, # 0 input res
            (16, 24, 40, 64, 112, 192, 320), # 1 channels
            (1, 3, 4, 2, 3, 4, 5), # 2 depths
            (1, 5, 5, 5, 6, 6, 6, 6), # 3 conv expansion ratio
            (3, 5, 5, 5, 5, 5, 5, 5), # 4 conv kr size
            (4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3), # 5 trans mlp ratio
            (4, 4, 7, 7, 7, 12, 12, 12, 12, 20, 20, 20, 20, 20), # 6 trans num heads
            (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), # 7 windows size
            (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), # 8 qk scale
            (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4) # 9 v scale
        )
        '''
        py = 0
        act = "swish"
        strides = (1, 2, 2, 2, 2, 1, 2) # didn't contain the first conv3x3
        use_se = (False, False, True,       # first three for MBlock stage,
                  False, True, True, True)  # and the last for Trans downsample layer

        # first_block
        hw = block_config[0]
        key = f"firstconv_{hw}_3_{block_config[1][0]}_2_3_{act}"
        py += self.predictor[key]
        if not self.silence: print(key)
        hw = hw // 2
        stage_cout = block_config[1][0]

        # layer_choice blocks
        conv_count, trans_count = 0, 0
        for stage_idx, channel in enumerate(block_config[1][1:-1]):
            name = "conv" if stage_idx <= 2 else "transformer"
            stage_stride = strides[stage_idx]
            stage_hwin = hw
            stage_hwout = hw // stage_stride
            hw = stage_hwout
            stage_cin = stage_cout
            stage_cout = channel
            if name == "conv":
                for i in range(block_config[2][stage_idx]):
                    s = stage_stride if i == 0 else 1
                    layer_hw = stage_hwin if i == 0 else stage_hwout
                    cin = stage_cin if i == 0 else stage_cout
                    cout = stage_cout
                    exp = block_config[3][conv_count]
                    ks = block_config[4][conv_count]
                    se = use_se[stage_idx]
                    conv_count += 1

                    # predict by lut
                    key = f"{name}_{layer_hw}_{cin}_{cout}_{exp}_{s}_{act}_{ks}_{'se' if se else 'nose'}"
                    py += self.predictor[key]
                    if not self.silence: print(key)

            elif name == "transformer":
                for i in range(block_config[2][stage_idx]):
                    s = stage_stride if i == 0 else 1
                    ds = "ds" if i == 0 else "nods"
                    layer_hw = stage_hwin if i == 0 else stage_hwout
                    cin = stage_cin if i == 0 else stage_cout
                    cout = stage_cout
                    exp = block_config[5][trans_count]
                    v = block_config[9][trans_count] if len(block_config) > 9 else 4
                    se = use_se[stage_idx]
                    trans_count += 1
                    
                    if self.mode == "layerwise": # predict by lut
                        ds_exp_mark = "_6" if i == 0 else ""
                        key = f"nasvit_{'se' if se else 'nose'}_transformer_{layer_hw}_{cin}_{cout}_{exp}_{s}_{act}_{v}_{ds}{ds_exp_mark}_{'ln' if self.layer_norm else 'bn'}"
                        lpy = self.predictor[key]
                        if not self.silence: print(key)

                        py += lpy

                    else: # predict by attn/ffn lut
                        tpy = 0

                        # downsample
                        if ds == "ds":
                            key = f"nasvit_{'se' if se else 'nose'}_transds_{layer_hw}_{cin}_{cout}_{s}_6"
                            tpy += self.predictor[key]
                            if not self.silence: print(key)
                            layer_hw = stage_hwout

                        # attn
                        key = f'nasvit_transattn_{layer_hw}_{cout}_{act}_{v}_{"ln" if self.layer_norm else "bn"}'
                        tpy += self.predictor[key]
                        if not self.silence: print(key)

                        # ffn
                        key = f'nasvit_transffn_{layer_hw}_{cout}_{exp}_{act}_{"ln" if self.layer_norm else "bn"}'
                        tpy += self.predictor[key]
                        if not self.silence: print(key)

                        py += tpy

            layer_hw = stage_hwout

        # MBPool block
        key = f"mbpool_{layer_hw}_{block_config[1][-2]}_{block_config[1][-1]}_6_{act}"
        py += self.predictor[key]
        if not self.silence: print(key)

        assert conv_count == len(block_config[4])
        assert trans_count == len(block_config[5])

        return py


    def get_single_block_arch(self, name, hw, cin, cout, kernel_size, expand_ratio, 
                    stride, activation):
        raise NotImplementedError # does not support latency predictor now
        block_type = self.get_type(name, cin, cout, stride, activation)
        dicts = get_block_arch_by_name(block_type, hw, cin, cout, kernel_size, expand_ratio, stride)
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
