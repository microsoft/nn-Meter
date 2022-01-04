# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from nn_meter.utils.graph_tool import ModelGraph
from .utils.constants import DUMMY_TYPES
from .utils.ir_tools import convert_nodes
from .rule_reader import RuleReader
from .rule_splitter import RuleSplitter


class KernelDetector:
    def __init__(self, rule_file):
        self.reader = RuleReader(rule_file)
        self.splitter = RuleSplitter(self.reader)
        self.model_graph = None
        self.bbs = []
        self._global_index = 0

    def load_graph(self, graph):
        new_graph = convert_nodes(graph)
        self.model_graph = ModelGraph(graph=new_graph)
        self.model_graph.refresh()
        self.bbs = self.splitter.split(self.model_graph)

    def get_kernels(self):
        kernels = []
        self._global_index = 0
        self._layer_kernel_dict = {}

        for bb in self.bbs:
            kernel = self._bb_to_kernel(bb)
            self._global_index += 1
            if kernel is not None:
                kernels.append(kernel)

        self._fetch_connections(kernels)
        return kernels

    def _fetch_connections(self, kernels):
        fusion_graph = self.splitter._fusion_graph

        for kernel in kernels:
            kernel["inbounds"] = []

        for i in range(len(fusion_graph)):
            layer = fusion_graph[i]
            kernel = self._layer_kernel_dict.get(layer)

            if kernel:
                outbounds = [fusion_graph.find_root(outbound) for outbound in fusion_graph.get_outbounds(i)]
                outbounds = [self._layer_kernel_dict[outbound] for outbound in outbounds]

                for outbound in outbounds:
                    outbound["inbounds"].append(kernel["name"])

                outbounds = [outbound["name"] for outbound in outbounds]
                kernel["outbounds"] = outbounds

    def _bb_to_kernel(self, bb):
        types = [self.model_graph.get_node_type(node) for node in bb]
        # logging.info(types)
        types = [t for t in types if t and t not in DUMMY_TYPES]

        if types:
            type = "-".join(types)
            name = f"{type}#{self._global_index}"

            kernel = {
                "op": type,
                "name": name,
            }

            layer = bb[0]
            self._layer_kernel_dict[layer] = kernel
            type = types[0]
            attr = self.model_graph.get_node_attr(layer)["attr"]
            input_shape = self.model_graph.get_node_attr(layer)["input_shape"]
            output_shape = self.model_graph.get_node_attr(layer)["output_shape"]

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

            if len(input_shape) >= 1:
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
