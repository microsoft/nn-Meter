sample = (
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

from nn_meter.predictor.transformer_predictor import BlockLatencyPredictor
predictor = BlockLatencyPredictor("pixel6_lut")
print(predictor.get_latency(sample))

# predictor = BlockLatencyPredictor("pixel4_lut")
# print(predictor.get_latency(sample))