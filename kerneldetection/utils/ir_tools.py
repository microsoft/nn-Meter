import copy
from .constants import *


def convert_nodes(graph):
    '''
    Resolve inconsistency between ONNX and Tensorflow
    '''
    new_graph = copy.deepcopy(graph)

    for _, node in new_graph.items():
        type = node['attr']['type']
        new_type = OP_ALIAS.get(type, type)
        node['attr']['type'] = new_type
        attr = node['attr']['attr']

        if 'kernel_shape' in attr:
            attr['ks'] = attr['kernel_shape']
            del attr['kernel_shape']

        if 'weight_shape' in attr and attr['weight_shape'] is not None:
            attr['ks'] = attr['weight_shape'][0:2]
            del attr['weight_shape']

        if 'ksize' in attr:
            attr['ks'] = attr['ksize']
            del attr['ksize']

        if new_type == 'split' and 'axis' in attr:
            attr['split_dim'] = attr['axis']
            del attr['split_dim']

    return new_graph
