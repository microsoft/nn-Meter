from ..common import make_divisible

class ProxylessNASSpace:
   
    def __init__(self, width_mult=1.0, num_classes=1000, hw=224):
        self.width_mult = width_mult
        self.ks_list = [3, 5, 7]
        self.expand_ratio_list = [3, 4, 6]
        self.depth_list = [2, 3, 4]

        base_stage_width = [32, 16, 24, 40, 80, 96, 192, 320, 1280]
        self.stage_width = [make_divisible(w * self.width_mult) for w in base_stage_width] 
        self.stride_stages = [2, 2, 2, 1, 2, 1]
        self.num_block_stages = [max(self.depth_list)] * 5 + [1]

        self.stage_hw = [hw // 2, hw // 2]
        for s in self.stride_stages:
            self.stage_hw.append(self.stage_hw[-1] // s)
        self.stage_hw.append(self.stage_hw[-1])

        self.block_configs = []
        self.block_configs.append(self.block_key('first_conv', (hw, self.stage_hw[0]), (3, self.stage_width[0])))
        self.block_configs.append(self.block_key('first_mbconv', (self.stage_hw[0], self.stage_hw[1]), (self.stage_width[0], self.stage_width[1]), k=3, e=1))
        for hwin, hwout, cin, cout in zip(self.stage_hw[1:-2], self.stage_hw[2:-1], 
                                          self.stage_width[1:-2], self.stage_width[2:-1]):
            for e in self.expand_ratio_list:
                for k in self.ks_list:
                    self.block_configs.append(self.block_key('mbconv', (hwin, hwout), (cin, cout), k=k, e=e))
            for e in self.expand_ratio_list:
                for k in self.ks_list:
                    self.block_configs.append(self.block_key('mbconv', (hwout, hwout), (cout, cout), k=k, e=e))

        self.block_configs.append(self.block_key('feature_mix', (self.stage_hw[-2], self.stage_hw[-1]), (self.stage_width[-2], self.stage_width[-1])))
        self.block_configs.append(self.block_key('logits', (self.stage_hw[-1], 1), (self.stage_width[-1], num_classes)))

    @staticmethod
    def block_key(name, hwio, cio, k=0, e=0):
        return dict(name=name, hwio=hwio, cio=cio, k=k, e=e)

    @staticmethod
    def config2str(config) -> str:
        hwin, hwout = [str(x) for x in config['hwio']]
        cin, cout = [str(x) for x in config['cio']]

        input_shape_str = f'input:{"x".join([hwin, hwin, cin])}'
        output_shape_str = f'output:{"x".join([hwout, hwout, cout])}'

        return '-'.join([config['name'], input_shape_str, output_shape_str, f'k:{config["k"]}', f'e:{config["e"]}'])
    
    @property
    def block_config_strs(self):
        try:
            return self._block_config_strs
        except AttributeError:
            self._block_config_strs = []
            for config in self.block_configs:
                self._block_config_strs.append(self.config2str(config))
            return self._block_config_strs