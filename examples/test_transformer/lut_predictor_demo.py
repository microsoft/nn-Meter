import os, json

base_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base_dir, "sampling.json"), 'r') as fp:
    sample = json.load(fp)

from nn_meter.predictor.transformer_predictor import BlockLatencyPredictor
predictor = BlockLatencyPredictor("mobile_lut")
print(predictor.get_latency(sample))