from turtle import width
from ..common import make_divisible

class MobileNetV3Space:
    
    def __init__(self, num_classes=1000, width_mult=1.0, hw=224):
        self.width_mult = width_mult
        self.ks_list = [3, 5, 7]
        self.expand_ratio_list = [3, 4, 6]
        self.depth_list = [2, 3, 4]
        self.hw = hw

        base_stage_width = [16, 16, 24, 40, 80, 112, 160, 960, 1280]
        self.stage_width = [make_divisible(w * self.width_mult) for w in base_stage_width] 
        self.stride_stages = [1, 2, 2, 2, 1, 2]
        self.act_stages = ['relu', 'relu', 'relu', 'h_swish', 'h_swish', 'h_swish']
        self.se_stages = [False, False, True, False, True, True]
        self.num_block_stages = [1] + [max(self.depth_list)] * 5

        self.block_configs = []
        self.add_config('first_conv', (self.hw, self.hw // 2), (3, self.stage_width[0]))
        self.add_config('first_mbconv', (self.hw // 2, self.hw // 2), (self.stage_width[0], self.stage_width[1]),
            k=3, e=1, act=self.act_stages[0], se=self.se_stages[0])

        hwin = self.hw // 2
        for cin, cout, act, se, strides in zip(self.stage_width[1:-3], self.stage_width[2:-2], 
            self.act_stages[1:], self.se_stages[1:], self.stride_stages[1:]):
            hwout = hwin // strides
            for e in self.expand_ratio_list:
                for k in self.ks_list:
                    self.add_config('mbconv', (hwin, hwout), (cin, cout), k=k, e=e, act=act, se=se)
            for e in self.expand_ratio_list:
                for k in self.ks_list:
                    self.add_config('mbconv', (hwout, hwout), (cout, cout), k=k, e=e, act=act, se=se)
            hwin = hwout
        
        self.add_config('final_expand', (hwin, hwin), (self.stage_width[-3], self.stage_width[-2]))
        self.add_config('feature_mix', (hwin, 1), (self.stage_width[-2], self.stage_width[-1]))
        self.add_config('logits', (1, 1), (self.stage_width[-1], num_classes))

    def add_config(self, name, hwio, cio, k=0, e=0, act='relu', se=0):
        self.block_configs.append(self.block_key(name, hwio, cio, k=k, e=e, act=act, se=int(se)))

    @staticmethod
    def block_key(name, hwio, cio, k=0, e=0, act='relu', se=0):
        return dict(name=name, hwio=hwio, cio=cio, k=k, e=e, act=act, se=se)

    @staticmethod
    def config2str(config) -> str:
        hwin, hwout = [str(x) for x in config['hwio']]
        cin, cout = [str(x) for x in config['cio']]

        input_shape_str = f'input:{"x".join([hwin, hwin, cin])}'
        output_shape_str = f'output:{"x".join([hwout, hwout, cout])}'

        return '-'.join([config['name'], input_shape_str, output_shape_str, 
            f'k:{config["k"]}', f'e:{config["e"]}', f'act:{config["act"]}', f'se:{int(config["se"])}'])
