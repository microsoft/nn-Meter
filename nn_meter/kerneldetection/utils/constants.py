# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
DUMMY_TYPES = [
    "Const",
    "Identity",
    "Placeholder",
]

# TODO: Refactor opset map. Should be moved to corresponding module.
OP_ALIAS = {
    # Tensorflow
    "Relu6": "relu",
    "Relu": "relu",
    "Add": "add",
    "Biasadd": "add",
    "Conv2D": "conv",
    "Reshape": "reshape",
    "FusedBatchNorm": "bn",
    "FusedBatchNormV3": "bn",
    "MatMul": "fc",
    "MaxPool": "maxpool",
    "AvgPool": "avgpool",
    "Mean": "gap",
    "Mul": "mul",
    "DepthwiseConv2dNative": "dwconv",
    "ConcatV2": "concat",
    "Split": "split",
    # ONNX
    "Conv": "conv",
    "BatchNormalization": "bn",
    "Slice": "split",
    "Concat": "concat",
    "AveragePool": "avgpool",
    "Relu": "relu",
    "Add": "add",
    "Gemm": "fc",
    "GlobalAveragePool": "gap",
    "Clip": "relu",
    "Mul": "mul",
    "Div": "div",
    "HardSigmoid": "hardsigmoid",
    "Flatten": "reshape",
    "Transpose": "transpose",
    "ReduceMean": "gap",
    "Split": "split",
    "Pad": "pad",
}
