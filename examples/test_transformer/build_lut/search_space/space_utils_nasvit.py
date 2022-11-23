
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

ACT = 'swish'
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
        'hw': [192, 224, 256, 288],
        'hw_out': [96, 112, 128, 144]
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
        'hw': [96, 112, 128, 144],
        'hw_out': [96, 112, 128, 144]
    },
    {
        'name': 'stage_1',
        'block_type': 0,
        'cin': [16, 24],
        'channel': [24, 32],
        'depth': [3, 4, 5],
        'expansion_ratio': [4, 5, 6],
        'kernel size': [5, 3],
        'stride': 2,
        'use_se': False,
        'hw': [96, 112, 128, 144],
        'hw_out': [48, 56, 64, 72]
    },
    {
        'name': 'stage_2', 
        'block_type': 0,  
        'cin': [24, 32],
        'channel': [32, 40],
        'depth': [3, 4, 5, 6],
        'expansion_ratio': [4, 5, 6],  
        'kernel size': [5, 3],    
        'stride': 2,
        'use_se': True,
        'hw': [48, 56, 64, 72],
        'hw_out': [24, 28, 32, 36]
    },
    {
        'name': 'stage_3', 
        'block_type': 1,  
        'cin': [32, 40],
        'channel': [64, 72],  
        'depth': [3, 4, 5, 6],  
        'expansion_ratio': [1, 2],
        'downsample_expansion_ratio': [6],
        'v_scale': [4],
        'stride': 2,          
        'hw': [24, 28, 32, 36],
        'hw_out': [12, 14, 16, 18]
    },
    {
        'name': 'stage_4', 
        'block_type': 1,  
        'cin': [64, 72],
        'channel': [112, 120, 128],
        'depth': [3, 4, 5, 6, 7, 8],  
        'expansion_ratio': [1, 2],  
        'downsample_expansion_ratio': [6],
        'v_scale': [4],
        'stride': 2,          
        'hw': [12, 14, 16, 18],
        'hw_out': [6, 7, 8, 9]
    },
    {
        'name': 'stage_5',
        'block_type': 1,
        'cin': [112, 120, 128],
        'channel': [160, 168, 176, 184],
        'depth': [3, 4, 5, 6, 7, 8],
        'expansion_ratio': [1, 2],
        'downsample_expansion_ratio': [6],
        'v_scale': [4],
        'stride': 1,
        'hw': [6, 7, 8, 9],
        'hw_out': [6, 7, 8, 9]
    },
    {
        'name': 'stage_6',
        'block_type': 1,
        'cin': [160, 168, 176, 184],
        'channel': [208, 216, 224],
        'depth': [3, 4, 5, 6],
        'expansion_ratio': [1, 2],
        'downsample_expansion_ratio': [6],
        'v_scale': [4],
        'stride': 2,
        'hw': [6, 7, 8, 9],
        'hw_out': [3, 4]
    },
    {
        'name': 'mb_pool',
        'block_type': 2,
        'cin': [208, 216, 224],
        'channel': [1792, 1984],
        'depth': None,
        'expansion_ratio': [6],
        'downsample_expansion_ratio': None,
        'v_scale': None,
        'stride': None,
        'hw': [3, 4],
        'hw_out': None
    }
]
