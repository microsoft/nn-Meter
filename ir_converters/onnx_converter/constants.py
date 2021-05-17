CONV_TYPE = 'Conv'
BN_TYPE = 'BatchNormalization'
SLICE_TYPE = 'Slice'
CONCAT_TYPE = 'Concat'
MAXPOOL_TYPE = 'MaxPool'
AVGPOOL_TYPE = 'AveragePool'
RELU_TYPE = 'Relu'
ADD_TYPE = 'Add'
FC_TYPE = 'Gemm'
RESHAPE_TYPE = 'Reshape'
GAP_TYPE = 'GlobalAveragePool'
CLIP_TYPE = 'Clip'
MUL_TYPE = 'Mul'
DIV_TYPE = 'Div'
HARDSIGMOID_TYPE = 'HardSigmoid'
FLATTEN_TYPE = 'Flatten'
TRANSPOSE_TYPE = 'Transpose'
REDUCEMEAN_TYPE = 'ReduceMean'
SPLIT_TYPE = 'Split'
PAD_TYPE = 'Pad'

OP_ALIAS = {
    CONV_TYPE: 'conv',
    BN_TYPE: 'bn',
    SLICE_TYPE: 'split',
    CONCAT_TYPE: 'concat',
    MAXPOOL_TYPE: 'maxpool',
    AVGPOOL_TYPE: 'avgpool',
    RELU_TYPE: 'relu',
    ADD_TYPE: 'add',
    FC_TYPE: 'fc',
    RESHAPE_TYPE: 'reshape',
    GAP_TYPE: 'gap',
    CLIP_TYPE: 'clip',
    MUL_TYPE: 'mul',
    DIV_TYPE: 'div',
    HARDSIGMOID_TYPE: 'hardsigmoid',
    FLATTEN_TYPE: 'flatten',
    TRANSPOSE_TYPE: 'transpose',
    REDUCEMEAN_TYPE: 'reducemean',
    SPLIT_TYPE: 'split',
    PAD_TYPE: 'pad',
}

ATTR_ALIAS = {
    'pads': ('padding', '__all__'),
    'axis': ('split_dim', ['split']),
}
