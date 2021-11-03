# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from nn_meter.utils.utils import try_import_onnx
import networkx as nx
from .utils import get_tensor_shape
from .constants import SLICE_TYPE
from itertools import chain
import logging


class OnnxConverter:
    def __init__(self, model):
        onnx = try_import_onnx()
        from onnx import shape_inference
        inferred_model = shape_inference.infer_shapes(model)
        self.graph = inferred_model.graph

        self.tensors = {}
        for tensor in chain(self.graph.input, self.graph.value_info, self.graph.output):
            self.tensors[tensor.name] = {
                "shape": get_tensor_shape(tensor),
                "inputs": [],
                "outputs": [],
            }

        for node in self.graph.node:
            for input_name in node.input:
                if input_name in self.tensors:
                    self.tensors[input_name]["outputs"].append(node)
            for output_name in node.output:
                if output_name in self.tensors:
                    self.tensors[output_name]["inputs"].append(node)

        self.G = self.to_networkx()

    def to_networkx(self):
        G = nx.DiGraph()

        sliced_tensors = set()
        selected_slice = set()
        for node in self.graph.node:
            if node.op_type == SLICE_TYPE:
                tensor = node.input[0]
                if tensor in sliced_tensors:
                    continue
                else:
                    sliced_tensors.add(tensor)
                    selected_slice.add(node.name)
            G.add_node(node.name, **self.fetch_attrs(node))

        for node in self.graph.node:
            if node.op_type == SLICE_TYPE and node.name not in selected_slice:
                continue
            for input_name in node.input:
                if input_name in self.tensors:  # remove dummy ops
                    G.add_edge(input_name, node.name)
            for output_name in node.output:
                if output_name in self.tensors:
                    G.add_edge(node.name, output_name)
                if node.op_type == SLICE_TYPE:
                    for tensor_name in self._get_sibling_slice_output_tensors(node):
                        G.add_edge(node.name, tensor_name)

        return G

    def fetch_attrs(self, node):
        from onnx import AttributeProto
        attrs = {}
        input_tensors = []
        for input_name in node.input:
            if input_name in self.tensors:
                input_tensors.append(self.tensors[input_name]["shape"])
        output_tensors = []
        for output_name in node.output:
            if output_name in self.tensors:
                output_tensors.append(self.tensors[output_name]["shape"])
        if node.op_type == SLICE_TYPE:
            for tensor_name in self._get_sibling_slice_output_tensors(node):
                output_tensors.append(self.tensors[tensor_name]["shape"])
        if (
            len(input_tensors) == 0
            or len(input_tensors[0]) <= 1
            or len(output_tensors) == 0
            or len(output_tensors[0]) <= 1
        ):
            logging.warning(f"Empty shape information with {node.name}")
            return attrs

        attrs["attr"] = {}
        attrs["type"] = node.op_type
        attrs["input_shape"] = input_tensors
        attrs["output_shape"] = output_tensors
        for attr in node.attribute:
            if attr.type == AttributeProto.FLOAT:
                attrs["attr"][attr.name] = attr.f
            elif attr.type == AttributeProto.INT:
                attrs["attr"][attr.name] = attr.i
            elif attr.type == AttributeProto.INTS:
                attrs["attr"][attr.name] = list(attr.ints)
            elif attr.type == AttributeProto.STRING:
                attrs["attr"][attr.name] = str(attr.s)
            else:
                logging.warning(f"Unsupported attributes type: {attr.type}")

        return attrs

    def convert(self):
        result = {}

        for node in self.G.nodes:
            node_attrs = self.G.nodes[node]
            if node in self.tensors or not node_attrs:
                continue

            outbounds = []
            inbounds = []
            for succ in self.G.successors(node):
                for succ_succ in self.G.successors(succ):
                    outbounds.append(succ_succ)
            for pred in self.G.predecessors(node):
                for pred_pred in self.G.predecessors(pred):
                    inbounds.append(pred_pred)

            result[node] = {
                "attr": node_attrs,
                "outbounds": outbounds,
                "inbounds": inbounds,
            }

        return result

    def _get_sibling_slice_output_tensors(self, node):
        output_tensors = []
        for slice in self.tensors[node.input[0]]["outputs"]:
            if slice.name != node.name and slice.op_type == SLICE_TYPE:
                for output_name in slice.output:
                    if output_name in self.tensors:
                        output_tensors.append(output_name)

        return output_tensors
