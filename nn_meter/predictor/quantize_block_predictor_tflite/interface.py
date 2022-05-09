# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from .. import load_latency_predictor
from ..quantize_block_predictor_onnx import BlockLatencyPredictor as ONNXPredictor

class BlockLatencyPredictor(ONNXPredictor):
    def __init__(self, predictor_name = "tflite27_cpu_int8"):
        self.predictor_name = predictor_name
        self.predictor = load_latency_predictor(predictor_name)
