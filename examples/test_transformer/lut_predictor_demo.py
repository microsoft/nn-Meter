sample1 = (
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

sample2 = [
    160, 
    (24, 24, 40, 48, 64, 160, 320), 
    (1, 2, 3, 1, 6, 3, 1), 
    (1, 2, 2, 4, 4, 4), 
    (3, 3, 3, 3, 3, 3), 
    (4, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1), 
    (3, 4, 4, 4, 4, 4, 4, 10, 10, 10, 20), 
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), 
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), 
    (2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 4)
]


NASVIT_A0 = (
    192, # res
    (16, 24, 32, 64, 112, 160, 208), # channels, 
    (1, 3, 3, 2, 2, 2, 2), # depths, 
    (1, 4, 4, 4, 4, 4, 4), # conv_ratio, 
    (3, 3, 3, 3, 3, 3, 3), # kr_size, 
    (1, 1, 1, 1, 1, 1, 1, 1), # mlp_ratio, 
    (8, 8, 14, 14, 20, 20, 26, 26), # num_heads, 
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), # window_size, 
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), # qk_scale, 
    (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4) # v_scale
)

from nn_meter.predictor.transformer_predictor import BlockLatencyPredictor

# predict latency with nasvit arch
predictor = BlockLatencyPredictor("pixel6_lut")
print(predictor.get_nasvit_latency(NASVIT_A0))

# # predict latency with our hybrid arch
# predictor = BlockLatencyPredictor("pixel6_lut")
# print(predictor.get_latency(NASVIT_A0))