
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
        'hw': [192, 224, 256, 288],
        'hw_out': [96, 112, 128, 144]
    },
    {
        'name': 'stage_0', 
        'block_type': 0, 
        'cin': [16],
        'channel': channel_list(start=24, end=32, step=8),  
        'depth': [3, 4, 5],  
        'expansion_ratio': [6, 4],  
        'kernel size': [5, 3],    
        'stride': 2,          
        'hw': [96, 112, 128, 144],
        'use_se': False,
        'hw_out': [48, 56, 64, 72]
    },
    {
        'name': 'stage_1', 
        'block_type': 0,  
        'cin': channel_list(start=24, end=32, step=8),
        'channel': channel_list(start=32, end=40, step=8),  
        'depth': [3, 4, 5, 6],  
        'expansion_ratio': [6, 4],  
        'kernel size': [5, 3],    
        'stride': 2,          
        'hw': [48, 56, 64, 72],
        'use_se': True,
        'hw_out': [24, 28, 32, 36]
    },
    {
        'name': 'stage_2', 
        'block_type': 0,  
        'cin': channel_list(start=32, end=40, step=8),
        'channel': channel_list(start=64, end=72, step=8),  
        'depth': [3, 4, 5, 6],  
        'expansion_ratio': [6, 4],  
        'kernel size': [5, 3],    
        'stride': 2,          
        'hw': [24, 28, 32, 36],
        'use_se': True,
        'hw_out': [12, 14, 16, 18]
    }, 
    {
        'name': 'stage_3', 
        'block_type': 1,  
        'cin': channel_list(start=64, end=72, step=8),
        'channel': channel_list(start=112, end=128, step=8),  
        'depth': [3, 4, 5, 6, 7, 8, 9],  
        'expansion_ratio': [4, 2],  
        'v_scale': [4],
        'stride': 2,          
        'hw': [12, 14, 16, 18],
        'hw_out': [6, 7, 8, 9]
    },
    {
        'name': 'stage_4', 
        'block_type': 1,  
        'cin': channel_list(start=112, end=128, step=8),
        'channel': channel_list(start=160, end=184, step=8),  
        'depth': [3, 4, 5, 6, 7, 8],  
        'expansion_ratio': [4, 2],  
        'v_scale': [4],
        'stride': 1,          
        'hw': [6, 7, 8, 9],
        'hw_out': [6, 7, 8, 9]
    },
    {
        'name': 'stage_5', 
        'block_type': 1,  
        'cin': channel_list(start=160, end=184, step=8),
        'channel': channel_list(start=208, end=224, step=8),  
        'depth': [3, 4, 5, 6],  
        'expansion_ratio': [4, 2],  
        'v_scale': [4],
        'stride': 2,          
        'hw': [6, 7, 8, 9],
        'hw_out': [3, 4, 5]
    }
]
