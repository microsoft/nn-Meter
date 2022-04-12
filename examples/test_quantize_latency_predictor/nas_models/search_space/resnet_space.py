from ..common import make_divisible


class ResNetSpace:

    BASE_DEPTH_LIST = [2, 2, 4, 2]
    STAGE_WIDTH_LIST = [256, 512, 1024, 2048]

    def __init__(self, num_classes=1000, hw=224) -> None:
        self.depth_list = [0, 1, 2]
        self.expand_ratio_list = [0.2, 0.25, 0.35]
        self.width_mult_list = [0.65, 0.8, 1.0]
        self.hw = hw

        self.num_blocks_list = [base_depth + max(self.depth_list) for base_depth in ResNetSpace.BASE_DEPTH_LIST]
        self.stride_list = [1, 2, 2, 2]

        self.stage_width_list = []
        for base_width in ResNetSpace.STAGE_WIDTH_LIST:
            self.stage_width_list.append(
                [make_divisible(base_width * width_mult) for width_mult in self.width_mult_list]
            )

        self.input_channel = [
            make_divisible(64 * width_mult) for width_mult in self.width_mult_list
        ]
        self.mid_input_channel = [
            make_divisible(c // 2) for c in self.input_channel
        ]

        input_channel = self.input_channel.copy()
        mid_input_channel = self.mid_input_channel.copy()
        
        self.block_configs = []
        for input_c in input_channel:
            for mid_input_c in mid_input_channel:
                for skipping in [0, 1]:
                    self.block_configs.append(dict(
                        name='input_stem', hwio=(hw, hw // 4), cio=(3, input_c), e=0, midc=mid_input_c, skipping=skipping
                    ))

        
        hwin = self.hw // 4
        for strides, width, in zip(self.stride_list, self.stage_width_list):
            hwout = hwin // strides
            for cin in input_channel:
                for cout in width:
                    for e in self.expand_ratio_list:
                        self.add_config('bconv', (hwin, hwout), (cin, cout), e)
            for cin in input_channel:
                for cout in width:
                    for e in self.expand_ratio_list:
                        self.add_config('bconv', (hwout, hwout), (cout, cout), e)
            input_channel = width
            hwin = hwout

        for cin in input_channel: 
            self.add_config('logits', (hwin, 1), (cin, num_classes))

    def add_config(self, name, hwio, cio, e=0, midc=0):
        self.block_configs.append(self.block_key(name, hwio, cio, e, midc))

    @staticmethod
    def block_key(name, hwio, cio, e=0, midc=0):
        return dict(name=name, hwio=hwio, cio=cio, e=e, midc=midc)

    @staticmethod
    def config2str(config) -> str:
        hwin, hwout = [str(x) for x in config['hwio']]
        cin, cout = [str(x) for x in config['cio']]

        input_shape_str = f'input:{"x".join([hwin, hwin, cin])}'
        output_shape_str = f'output:{"x".join([hwout, hwout, cout])}'

        config_str = '-'.join([config['name'], input_shape_str, output_shape_str, f'e:{config["e"]}', f'midc:{config["midc"]}'])
        if config['name'] == 'input_stem':
            config_str += f'-skipping:{config["skipping"]}'

        return config_str

    @property
    def block_config_strs(self):
        try:
            return self._block_config_strs
        except AttributeError:
            self._block_config_strs = []
            for config in self.block_configs:
                self._block_config_strs.append(self.config2str(config))
            return self._block_config_strs