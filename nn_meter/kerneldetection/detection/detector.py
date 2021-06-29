# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from nn_meter.kerneldetection.rulelib.rule_reader import RuleReader
from nn_meter.kerneldetection.rulelib.rule_splitter import RuleSplitter
from nn_meter.utils.graphe_tool import Graphe
from nn_meter.kerneldetection.utils.constants import DUMMY_TYPES
from nn_meter.kerneldetection.utils.ir_tools import convert_nodes


class KernelDetector:
    def __init__(self, rule_file):
        self.reader = RuleReader(rule_file)
        self.splitter = RuleSplitter(self.reader)
        self.graph = None
        self.bbs = []

    def load_graph(self, graph):
        new_graph = convert_nodes(graph)
        self.graph = Graphe(graph=new_graph)
        self.graph.refresh()
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
        # print(types)
        types = [t for t in types if t and t not in DUMMY_TYPES]

        if types:
            type = "-".join(types)

            kernel = {
                "op": type,
            }

            layer = bb[0]
            type = types[0]
            attr = self.graph.get_node_attr(layer)["attr"]
            input_shape = self.graph.get_node_attr(layer)["input_shape"]
            output_shape = self.graph.get_node_attr(layer)["output_shape"]

            # Remove const from first biasadd of hswish
            if type == "hswish":
                input_shape = [input_shape[0]]
            kernel["input_tensors"] = input_shape

            if "ks" in attr:
                kernel["ks"] = attr["ks"]
            if "strides" in attr:
                kernel["strides"] = attr["strides"]
            if "split_dim" in attr:
                kernel["split_dim"] = attr["split_dim"]

            if len(input_shape) == 1:
                if len(input_shape[0]) == 4:
                    kernel["inputh"] = input_shape[0][1]
                    kernel["inputw"] = input_shape[0][2]
                kernel["cin"] = input_shape[0][-1]

            if len(output_shape) == 1:
                kernel["cout"] = output_shape[0][-1]
            elif len(output_shape) > 1:
                kernel["output_tensors"] = output_shape

            return kernel
        else:
            return None
