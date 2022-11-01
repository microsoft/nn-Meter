
def channel_list(start, end, step=8):
    return [c for c in range(start, end+1, step)]

'''
    block type: 0 conv, 1 transformer
    conv layer: conv_block.py -> function ``conv_layer``
    transformer layer: transformer_block.py -> function ``transformer_layer``

    for all conv layer, ``se`` option is disabled.

    ``channel`` is the ***output channel***
    ``hw`` is the ***input resolution***
'''
reproduce_nasvit = True
if reproduce_nasvit:
    extend_width = [
        [72],
        [120],
        [168, 184],
        [216],
        [1792]
    ]
else:
    extend_width = [[], [], [], [], []]

ACT = 'hard_swish'
configs = [
    {
        'name': 'first_conv',
        'block_type': -1,
        'cin': [3],
        'channel': [16, 24],
        'depth': [1],
        'expansion_ratio': [1],
        'kernel size': [3],
        'stride': 2,
        'hw': [128, 144, 160, 176, 192, 224, 256, 288],
        'hw_out': [64, 72, 80, 88, 96, 112, 128, 144]
    },
    {
        'name': 'stage_0',
        'block_type': 0,
        'cin': [16, 24],
        'channel': [16, 24],
        'depth': [1, 2],
        'expansion_ratio': [1],
        'kernel size': [5, 3],
        'stride': 1,
        'use_se': False,
        'hw': [64, 72, 80, 88, 96, 112, 128, 144],
        'hw_out': [64, 72, 80, 88, 96, 112, 128, 144]
    },
    {
        'name': 'stage_1',
        'block_type': 0,
        'cin': [16, 24],
        'channel': channel_list(start=16, end=32, step=8),
        'depth': [1, 2, 3, 4, 5],
        'expansion_ratio': [2, 3, 4, 5, 6],
        'kernel size': [5, 3],
        'stride': 2,
        'use_se': False,
        'hw': [64, 72, 80, 88, 96, 112, 128, 144],
        'hw_out': [32, 36, 40, 44, 48, 56, 64, 72]
    },
    {
        'name': 'stage_2', 
        'block_type': 0,  
        'cin': channel_list(start=16, end=32, step=8),
        'channel': channel_list(start=16, end=40, step=8),
        'depth': [1, 2, 3, 4, 5, 6],
        'expansion_ratio': [2, 3, 4, 5, 6],  
        'kernel size': [5, 3],    
        'stride': 2,
        'use_se': True,
        'hw': [32, 36, 40, 44, 48, 56, 64, 72],
        'hw_out': [16, 18, 20, 22, 24, 28, 32, 36]
    },
    {
        'name': 'stage_3', 
        'block_type': 1,  
        'cin': channel_list(start=16, end=40, step=8),
        'channel': channel_list(start=32, end=96, step=16) + extend_width[0],  
        'depth': [1, 2, 3, 4, 5, 6],  
        'expansion_ratio': [1, 2, 3, 4],
        'downsample_expansion_ratio': [4, 6],
        'v_scale': [1, 2, 3, 4, 5, 6],
        'stride': 2,          
        'hw': [16, 18, 20, 22, 24, 28, 32, 36],
        'hw_out': [8, 9, 10, 11, 12, 14, 16, 18]
    },
    {
        'name': 'stage_4', 
        'block_type': 1,  
        'cin': channel_list(start=32, end=96, step=16) + extend_width[0],
        'channel': channel_list(start=48, end=160, step=16) + extend_width[1],
        'depth': [1, 2, 3, 4, 5, 6, 7, 8],  
        'expansion_ratio': [1, 2, 3, 4],  
        'downsample_expansion_ratio': [4, 6],
        'v_scale': [1, 2, 3, 4],
        'stride': 2,          
        'hw': [8, 9, 10, 11, 12, 14, 16, 18],
        'hw_out': [4, 5, 6, 7, 8, 9]
    },
    {
        'name': 'stage_5',
        'block_type': 1,
        'cin': channel_list(start=48, end=160, step=16) + extend_width[1],
        'channel': channel_list(start=96, end=256, step=16) + extend_width[2],
        'depth': [1, 2, 3, 4, 5, 6, 7, 8],
        'expansion_ratio': [1, 2, 3, 4],
        'downsample_expansion_ratio': [4, 6],
        'v_scale': [1, 2, 3, 4],
        'stride': 1,
        'hw': [4, 5, 6, 7, 8, 9],
        'hw_out': [4, 5, 6, 7, 8, 9]
    },
    {
        'name': 'stage_6',
        'block_type': 1,
        'cin': channel_list(start=96, end=256, step=16) + extend_width[2],
        'channel': channel_list(start=144, end=320, step=16) + extend_width[3],
        'depth': [1, 2, 3, 4, 5, 6],
        'expansion_ratio': [1, 2, 3, 4],
        'downsample_expansion_ratio': [4, 6],
        'v_scale': [1, 2, 3, 4],
        'stride': 2,
        'hw': [4, 5, 6, 7, 8, 9],
        'hw_out': [2, 3, 4]
    },
    {
        'name': 'mb_pool',
        'block_type': 2,
        'cin': channel_list(start=144, end=320, step=16) + extend_width[3],
        'channel': [1984] + extend_width[4],
        'depth': None,
        'expansion_ratio': [6],
        'downsample_expansion_ratio': None,
        'v_scale': None,
        'stride': None,
        'hw': [2, 3, 4],
        'hw_out': None
    }
]
