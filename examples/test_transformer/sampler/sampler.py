import random
import math

#bignas space:
SPACE = [
    [
        [3, 3, 4, 6, 6, 6], # MAX_DEPTH
        [2, 2, 2, 2, 2, 3], # MIN_DEPTH

        [32, 48, 88, 128, 216, 352], # MAX_CHANNELS
        [24, 40, 80, 112, 192, 320], # MIN_CHANNELS

        [2, 4, 6],

        [2, 1],

        [4, 6, 8],

        [[7, 14], [7], [1]],

        [[2.5, 2, 1.5, 1], [2, 1], [1]],
        [[1, 0.6], [1, 0.6], [1]],
    ],

    [
        [3, 3, 4, 6, 6, 2],
        [2, 2, 2, 2, 2, 1],

        [32, 48, 88, 96, 184, 320],
        [24, 40, 80, 88, 160, 288],

        [2, 4, 6],

        [2, 1],

        [4, 6, 8],

        [[7, 14], [7], [1]],

        [[2.5, 2, 1.5, 1], [2, 1], [1]],
        [[1, 0.6], [1, 0.6], [1]],
    ],

    [
        [3, 3, 4, 6, 6, 2],
        [2, 2, 2, 2, 2, 1],

        [32, 48, 88, 160, 248, 384],
        [24, 40, 80, 144, 224, 352],

        [2, 4, 6],

        [2, 1],

        [4, 6, 8],

        [[7, 14], [7], [1]],

        [[2.5, 2, 1.5, 1], [2, 1], [1]],
        [[1, 0.6], [1, 0.6], [1]],
    ]
]

STAGE = ['C', 'C', 'C', 'T', 'T', 'T'] 
RESOLUTION = [160, 192, 224, 256]
DOWNSAMPLING = [True, True, True, False, True, True]

MAX_DEPTH = SPACE[0][0]
MIN_DEPTH = SPACE[0][1]

MAX_CHANNELS = SPACE[0][2]
MIN_CHANNELS = SPACE[0][3]

CONV_RATIO = [4, 6]

MLP_RATIO = [2, 1]

HEADS = [4, 6, 8]

WINDOWS_SIZE = [[7, 14], [7], [1]]

QK_SCALE = [[2.5, 2, 1.5, 1], [2, 1], [1]]
V_SCALE = [[1, 0.6], [1, 0.6], [1]]

ACT = 'hard_swish'
LAYER_NORM = False

def arch_sampling(mode='uniform', override_channels='', override_depths='', override_conv_ratio='', override_kr_size='', override_mlp_ratio='', 
                    override_num_heads='', override_qk_scale='', override_v_scale='', override_windows_size=''):
    channels = []

    resolution = random.choice(RESOLUTION)

    dict = {}

    feature_map_size = resolution // 2 # stem
    for index, s in enumerate(STAGE):
        stage = 'conv' if s == 'C' else 'transformer'
        
        if stage == 'conv':
            if DOWNSAMPLING[index]:
                feature_map_size = feature_map_size // 2
                
            stage_info = {
                'input_channel': 0,
                'output_channel': 0,
                'downsampling': DOWNSAMPLING[index],
                'input_feature_map_size': feature_map_size * 2 if DOWNSAMPLING[index] else feature_map_size,
                'feature_map_size': feature_map_size, 
                'depth': 0,
                'kernel_size': [],
                'conv_ratio': []
            }
        else:
            if DOWNSAMPLING[index]:
                feature_map_size = math.ceil(feature_map_size/2)
            
            stage_info = {
                'input_channel': 0,
                'output_channel': 0,
                'downsampling': DOWNSAMPLING[index],
                'input_feature_map_size': feature_map_size * 2 if DOWNSAMPLING[index] else feature_map_size,
                'feature_map_size': feature_map_size, 
                'depth': 0,
                'window_size': [],
                'mlp_ratio': [],
                'num_heads': [],
                'qk_scale': [],
                'v_scale': []
            }
        
        dict[f'stage_{index}_type_{stage}'] = stage_info
    
    # macro arch, i.e., channels and depths
    while True:
        if len(channels) == len(MAX_CHANNELS):
            break

        idx = len(channels)
        if mode == 'min':
            channel = MIN_CHANNELS[idx]
        elif mode == 'max':
            channel = MAX_CHANNELS[idx]
        elif mode == 'uniform':
            channel = random.randint(MIN_CHANNELS[idx]//8, MAX_CHANNELS[idx]//8) * 8
        if channel % 8 == 0:
            channels.append(channel)
        else:
            print('Channels % 8 != 0!')

    if mode == 'max':
        depths = [MAX_DEPTH[i] for i in range(len(MAX_DEPTH))]
    elif mode == 'min':
        depths = [MIN_DEPTH[i] for i in range(len(MIN_DEPTH))]
    elif mode == 'uniform':
        depths = [random.randint(MIN_DEPTH[i], MAX_DEPTH[i]) for i in range(len(MAX_DEPTH))]
    
    input_channel = 16
    for index, (k, v) in enumerate(dict.items()):
        v['input_channel'] = input_channel
        v['output_channel'] = channels[index]
        v['depth'] = depths[index]
        input_channel = channels[index]
    
    num_conv_stage = STAGE.count('C')
    num_conv_layers = sum(depths[:num_conv_stage])

    for index, (k, v) in enumerate(dict.items()):

        if index < num_conv_stage:
            # for conv layers
            if mode == 'max':
                conv_ratio = [max(CONV_RATIO) for _ in range(depths[index])]
                kr_size = [5 for _ in range(depths[index])]
            elif mode == 'min':
                conv_ratio = [min(CONV_RATIO) for _ in range(depths[index])]
                kr_size = [3 for _ in range(depths[index])]
            elif mode == 'uniform':
                conv_ratio = [random.choice(CONV_RATIO) for _ in range(depths[index])]
                kr_size = [random.choice([3, 5]) for _ in range(depths[index])]
            v['conv_ratio'] = conv_ratio
            v['kernel_size'] = kr_size
        else:
            feature_map_size = v['feature_map_size']
            # for transformer layers

            def is_legal_window_size(window_size, resample=False):
                if window_size >= feature_map_size or feature_map_size % window_size != 0:
                    if not resample:
                        return 1
                    else:
                        min_window_size = min(WINDOWS_SIZE[index-num_conv_stage])

                        if min_window_size >= feature_map_size or feature_map_size % min_window_size != 0:
                            return 1
                        else:
                            return is_legal_window_size(random.choice(WINDOWS_SIZE[index-num_conv_stage]), resample = True)
                else:
                    return window_size
            
            if mode == 'max':
                mlp_ratio = [max(MLP_RATIO) for _ in range(depths[index])]
                num_heads = [max(HEADS) for _ in range(depths[index])]
                qk_scale = [min(QK_SCALE[index-num_conv_stage]) for _ in range(depths[index])]
                v_scale = [max(V_SCALE[index-num_conv_stage]) for _ in range(depths[index])]
                window_size = [is_legal_window_size(max(WINDOWS_SIZE[index-num_conv_stage])) for _ in range(depths[index])]
            elif mode == 'min':
                mlp_ratio = [min(MLP_RATIO) for _ in range(depths[index])]
                num_heads = [min(HEADS) for _ in range(depths[index])]
                qk_scale = [max(QK_SCALE[index-num_conv_stage]) for _ in range(depths[index])]
                v_scale = [min(V_SCALE[index-num_conv_stage]) for _ in range(depths[index])]
                window_size = [is_legal_window_size(min(WINDOWS_SIZE[index-num_conv_stage])) for _ in range(depths[index])]
            elif mode == 'uniform':
                mlp_ratio = [random.choice(MLP_RATIO) for _ in range(depths[index])]
                num_heads = [random.choice(HEADS) for _ in range(depths[index])]
                qk_scale = [random.choice(QK_SCALE[index-num_conv_stage]) for _ in range(depths[index])]
                v_scale = [random.choice(V_SCALE[index-num_conv_stage]) for _ in range(depths[index])]
                window_size = [is_legal_window_size(random.choice(WINDOWS_SIZE[index-num_conv_stage]), resample=True) for _ in range(depths[index])]
            
            v['mlp_ratio'] = mlp_ratio
            v['num_heads'] = num_heads
            v['qk_scale'] = qk_scale
            v['v_scale'] = v_scale
            v['window_size'] = window_size
    
    dict['input_image_resolution'] = resolution
    return dict

def arch_sampling_unpacked(mode):
    dict = arch_sampling(mode)
    with open("/data1/jiahang/working/pixel6_fp32_workspace/predictor_build/results/sampling.txt", "w") as fp:
        import json
        json.dump(dict, fp, indent=4)

    num_conv_stage = STAGE.count('C')
    num_transformer_stage = STAGE.count('T')

    channels, depths, conv_ratio, kr_size, mlp_ratio, num_heads, window_size, qk_scale, v_scale = [], [], [], [], [], [], [], [], []
    for index, (_, v) in enumerate(dict.items()):
        if index < num_conv_stage + num_conv_stage:
            depths += [v['depth']]
            channels += [v['output_channel']]

            if index < num_conv_stage:
                kr_size += v['kernel_size']
                conv_ratio += v['conv_ratio']
            else:
                mlp_ratio += v['mlp_ratio']
                num_heads += v['num_heads']
                window_size += v['window_size']
                qk_scale += v['qk_scale']
                v_scale += v['v_scale']
    
    return dict['input_image_resolution'], channels, depths, conv_ratio, kr_size, mlp_ratio, num_heads, window_size, qk_scale, v_scale


if __name__ == '__main__':
    # N = 1
    # for i in range(N):
    #     ret = arch_sampling()
    #     with open("/data1/jiahang/working/pixel6_fp32_workspace/predictor_build/results/sampling.txt", "w") as fp:
    #         import json
    #         json.dump(ret, fp, indent=4)
    #     # print(ret)
    print(arch_sampling_unpacked(mode='uniform'))