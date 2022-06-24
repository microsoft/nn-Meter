from nn_meter.predictor.quantize_block_predictor import BlockLatencyPredictor
pred = BlockLatencyPredictor("tflite27_cpu_int8")
block_list = [
    {
        "name": "MobileNetV3ResBlock",
        "hw": 112,
        "cin": 16,
        "cout": 32,
        "kernel_size": 3,
        "expand_ratio": 0.5,
        "stride": 1,
        "activation": "swish"
    },
    {
        "name": "MobileNetV3K3ResBlock",
        "hw": 112,
        "cin": 16,
        "cout": 32,
        "kernel_size": 3,
        "expand_ratio": 0.5,
        "stride": 1,
        "activation": "swish"
    },
    # {}
]
print(pred.get_latency(block_list))