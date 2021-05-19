DUMMY_TYPES = [
    'Const',
    'Identity',
    'Placeholder',
]

TENSORFLOW_OP_ALIAS = {
    'Relu6': 'relu',
    'Relu': 'relu',
    'Add': 'add',
    'Biasadd': 'add',
    'Conv2D': 'conv',
    'Reshape': 'reshape',
    'FusedBatchNorm': 'bn',
    'FusedBatchNormV3': 'bn',
    'MatMul': 'fc',
    'MaxPool': 'maxpool',
    'AvgPool': 'avgpool',
    'Mean': 'gap',
    'Mul': 'mul',
    'DepthwiseConv2dNative': 'dwconv',
    'ConcatV2': 'concat',
    'Split': 'split',
}
