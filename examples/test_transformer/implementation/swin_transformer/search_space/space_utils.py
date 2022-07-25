
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
        'cin': [3],
        'channel': [16],  
        'depth': [1],  
        'expansion_ratio': [1],  
        'kernel size': [3],    
        'stride': 2,          
        'hw': [176, 192, 208, 224],
        'hw_out': [88, 96, 104, 112]
    },
    {
        'name': 'stage_0', 
        'block_type': 0, 
        'cin': [16],
        'channel': channel_list(start=24, end=32, step=8),  
        'depth': [2, 3],  
        'expansion_ratio': [6, 4],  
        'kernel size': [5, 3],    
        'stride': 2,          
        'hw': [88, 96, 104, 112],
        'hw_out': [44, 48, 52, 56]
    },
    {
        'name': 'stage_1', 
        'block_type': 0, 
        'cin': channel_list(start=24, end=32, step=8), 
        'channel': channel_list(start=40, end=48, step=8),  
        'depth': [2, 3],  
        'expansion_ratio': [6, 4],  
        'kernel size': [5, 3],    
        'stride': 2,          
        'hw': [44, 48, 52, 56],
        'hw_out': [22, 24, 26, 28]
    },
    {
        'name': 'stage_2', 
        'block_type': 0, 
        'cin': channel_list(start=40, end=48, step=8),  
        'channel': channel_list(start=80, end=88, step=8),
        'depth': [2, 3, 4, 5, 6],  
        'expansion_ratio': [6, 4],  
        'kernel size': [5, 3],    
        'stride': 2,          
        'hw': [22, 24, 26, 28],
        'hw_out': [11, 12, 13, 14]
    }, 
    {
        'name': 'stage_3', 
        'block_type': 1, 
        'cin': channel_list(start=80, end=88, step=8),  
        'channel': channel_list(start=112, end=128, step=8),  
        'depth': [2, 3, 4, 5, 6],  
        'expansion_ratio': [4, 2],  
        'v_scale': [4, 2],
        'stride': 1,          
        'hw': [11, 12, 13, 14],
        'hw_out': [11, 12, 13, 14]
    },
    {
        'name': 'stage_4', 
        'block_type': 1, 
        'cin': channel_list(start=112, end=128, step=8), 
        'channel': channel_list(start=192, end=216, step=8),  
        'depth': [3, 4, 5, 6, 7, 8],  
        'expansion_ratio': [4, 2],  
        'v_scale': [4, 2],
        'stride': 2,          
        'hw': [11, 12, 13, 14],
        'hw_out': [6, 7]
    },
    {
        'name': 'stage_5', 
        'block_type': 1, 
        'cin': channel_list(start=192, end=216, step=8),  
        'channel': channel_list(start=320, end=352, step=8),  
        'depth': [3, 4, 5, 6, 7, 8],  
        'expansion_ratio': [4, 2],  
        'v_scale': [4, 2],
        'stride': 2,          
        'hw': [6, 7],
        'hw_out': [3, 4]
    }
]
