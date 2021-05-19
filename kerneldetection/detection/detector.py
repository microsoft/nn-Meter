from kerneldetection.rulelib.rule_reader import RuleReader
from kerneldetection.rulelib.rule_splitter import RuleSplitter
from utils.grapher_tool import Grapher
from .constants import DUMMY_TYPES


class KernelDetector:
    def __init__(self, rule_file):
        self.reader = RuleReader(rule_file)
        self.splitter = RuleSplitter(self.reader)
        self.graph = None
        self.bbs = []

    def load_graph(self, graph):
        self.graph = Grapher(graph=graph)
        self.bbs = self.splitter.split(self.graph)

    @property
    def kernels(self):
        kernels = []

        for bb in self.bbs:
            kernel = self._bb_to_kernel(bb)
            if kernel is not None:
                kernels.append(kernel)

        return kernels

    def _bb_to_kernel(self, bb):
        types = [self.graph.get_node_type(node) for node in bb]
        #print(types)
        types = [t for t in types if t and t not in DUMMY_TYPES]

        if types:
            type = '-'.join(types)

            kernel = {
                'op': type,
            }

            layer = bb[0]
            type = types[0]
            attr = self.graph.get_node_attr(layer)['attr']
            input_shape = self.graph.get_node_attr(layer)['input_shape']
            output_shape = self.graph.get_node_attr(layer)['output_shape']
            if type in ['conv', 'dwconv']:
                kernel['ks'] = attr['ks']
                kernel['cin'] = input_shape[0][3]
                kernel['cout'] = output_shape[0][3]
                kernel['strides'] = attr['strides']
                if type == 'dwconv':
                    kernel['cout'] = kernel['cin']
            elif type in ['maxpool', 'avgpool']:
                kernel['ks'] = attr['ksize']
                kernel['cin'] = input_shape[0][3]
                kernel['cout'] = output_shape[0][3]
                kernel['strides'] = attr['strides']
            elif type == 'fc':
                kernel['cin'] = input_shape[0][1]
                kernel['cout'] = output_shape[0][1]
            elif type == 'gap':
                kernel['cin'] = input_shape[0][3]
                kernel['cout'] = output_shape[0][3]
            elif type in ['relu','hswish']:
                kernel['cin'] = input_shape[-1]
                kernel['cout'] = output_shape[-1]

            kernel['input_tensors'] = input_shape
            if type not in ['relu','bn', 'fc', 'reshape',  'Pack', 'StridedSlice','split']:
                kernel['inputh'] = input_shape[0][1]
                kernel['inputw'] = input_shape[0][2]

            if type == 'split':
                kernel['split_dim'] = attr['split_dim']
                kernel['output_tensors'] = output_shape

            return kernel
        else:
            return None
