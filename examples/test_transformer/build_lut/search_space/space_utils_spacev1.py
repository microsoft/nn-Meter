
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

ACT = 'hard_swish'
configs = [
    {
        'name': 'first_conv', 
        'block_type': 0, 
        'channel': [16],  
        'depth': [1],  
        'expansion_ratio': [1],  
        'kernel size': [3],    
        'stride': 2,          
        'hw': [176, 192, 224],
        'hw_out': [88, 96, 112]
    },
    {
        'name': 'stage_0', 
        'block_type': 0,  
        'cin': [16],
        'channel': channel_list(start=24, end=32, step=8),  
        'depth': [2, 3, 4, 5],  
        'expansion_ratio': [6, 4, 3],  
        'kernel size': [5, 3],    
        'stride': 2,          
        'hw': [88, 96, 112],
        'hw_out': [44, 48, 56],
        'use_se': False
    },
    {
        'name': 'stage_1', 
        'block_type': 0,  
        'cin': channel_list(start=24, end=32, step=8),
        'channel': channel_list(start=32, end=40, step=8),  
        'depth': [2, 3, 4, 5, 6],  
        'expansion_ratio': [6, 4, 3],  
        'kernel size': [5, 3],    
        'stride': 2,          
        'hw': [44, 48, 56],
        'hw_out': [22, 24, 28],
        'use_se': True
    },
    {
        'name': 'stage_2', 
        'block_type': 0,  
        'cin': channel_list(start=32, end=40, step=8),
        'channel': channel_list(start=64, end=72, step=8),  
        'depth': [2, 3, 4, 5, 6],  
        'expansion_ratio': [6, 4, 3],  
        'kernel size': [5, 3],    
        'stride': 2,          
        'hw': [22, 24, 28],
        'hw_out': [11, 12, 14],
        'use_se': True
    }, 
    {
        'name': 'stage_3', 
        'block_type': 1,  
        'cin': channel_list(start=64, end=72, step=8),
        'channel': channel_list(start=128, end=160, step=8),  
        'depth': [2, 3, 4, 5, 6, 7, 8],  
        'expansion_ratio': [4, 2, 1],  
        'v_scale': [4, 2],
        'stride': 1,          
        'hw': [11, 12, 14],
        'hw_out': [11, 12, 14],
    },
    {
        'name': 'stage_4', 
        'block_type': 1,  
        'cin': channel_list(start=128, end=160, step=8),
        'channel': channel_list(start=208, end=248, step=8),  
        'depth': [3, 4, 5, 6, 7, 8],  
        'expansion_ratio': [4, 2, 1],  
        'v_scale': [4, 2],
        'stride': 2,          
        'hw': [11, 12, 14],
        'hw_out': [6, 7],
    },
    {
        'name': 'stage_5', 
        'block_type': 1,   
        'cin': channel_list(start=208, end=248, step=8),
        'channel': channel_list(start=272, end=320, step=8),  
        'depth': [3, 4, 5, 6],  
        'expansion_ratio': [4, 2, 1],  
        'v_scale': [4, 2],
        'stride': 2,          
        'hw': [6, 7],
        'hw_out': [3, 4],
    }
]
