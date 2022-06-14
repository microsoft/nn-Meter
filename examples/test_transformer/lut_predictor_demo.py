sample = (
    176, # input res
    (0, 0, 0, 1, 1, 1), # block type
    (24, 40, 80, 112, 192, 320), # channels
    (2, 3, 2, 2, 3, 4), # depths
    (6, 6, 6, 6, 6, 4, 4), # conv expansion ratio
    (3, 3, 3, 3, 3, 5, 5), # conv kr size
    (2, 2, 4, 4, 4, 4, 4, 4, 4), # trans mlp ratio
    (14, 14, 24, 24, 24, 40, 40, 40, 40), # trans num heads
    (1, 1, 1, 1, 1, 1, 1, 1, 1), # windows size
    (1, 1, 1, 1, 1, 1, 1, 1, 1), # qk scale
    (2, 2, 2, 2, 2, 2, 2, 2, 2) # v scale
)

from nn_meter.predictor.transformer_predictor import BlockLatencyPredictor
predictor = BlockLatencyPredictor("mobile_lut")
print(predictor.get_latency(sample))