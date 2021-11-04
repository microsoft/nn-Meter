# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import tempfile
from ..onnx_converter import OnnxConverter
from .opset_map import nni_attr_map, nni_type_map
from nn_meter.utils.import_package import try_import_onnx, try_import_torch, try_import_onnxsim, try_import_nni


def _nchw_to_nhwc(shapes):
    return [
        [shape[0], shape[2], shape[3], shape[1]]
        if len(shape) == 4
        else shape
        for shape in shapes
    ]


class NNIIRConverter:
    def __init__(self, ir_model):
        try_import_nni()
        try:
            from nni.retiarii.converter.utils import flatten_model_graph
            self.ir_model = flatten_model_graph(ir_model)
        except:
            from nni.retiarii.converter.graph_gen import GraphConverterWithShape
            self.ir_model = ir_model.fork()
            GraphConverterWithShape().flatten(self.ir_model)

    def convert(self):
        graph = self._to_graph_layout()

        for _, node in graph.items():
            self._map_opset(node)

        self._remove_unshaped_nodes(graph)

        return graph

    def _to_graph_layout(self):
        graph = {}

        for node in self.ir_model.root_graph.hidden_nodes:
            node_dict = {
                "attr": {
                    "attr": {
                        k: v
                        for k, v in node.operation.parameters.items()
                    },
                    "input_shape": _nchw_to_nhwc(node.operation.parameters.get("input_shape")
                                                 if "input_shape" in node.operation.parameters 
                                                 else node.operation.attributes.get('input_shape')),
                    "output_shape": _nchw_to_nhwc(node.operation.parameters.get("output_shape") 
                                                 if "output_shape" in node.operation.parameters 
                                                 else node.operation.attributes.get('output_shape')),
                    "type": node.operation.type,
                },
                "inbounds": [],
                "outbounds": [],
            }

            incoming_edges = sorted(node.incoming_edges, key=lambda e: e.tail_slot or 0)
            for edge in incoming_edges:
                node_dict["inbounds"].append(edge.head.name)

            outgoing_edges = sorted(node.outgoing_edges, key=lambda e: e.head_slot or 0)
            for edge in outgoing_edges:
                node_dict["outbounds"].append(edge.tail.name)

            graph[node.name] = node_dict

        return graph

    def _map_opset(self, node):
        old_type = node["attr"]["type"]
        new_type = nni_type_map.get(old_type, old_type)

        new_attr_dict = {}
        for attr_name, attr_value in node["attr"]["attr"].items():
            new_attr_name = attr_name
            new_attr_value = attr_value
            for type, attr_map in nni_attr_map.items():
                if type == "__all__" or type == new_type:
                    if attr_name in attr_map:
                        new_attr_name, modifier = attr_map[attr_name]
                        if modifier is not None:
                            new_attr_value = modifier(attr_value)

            new_attr_dict[new_attr_name] = new_attr_value

        node["attr"]["type"] = new_type
        node["attr"]["attr"] = new_attr_dict

    def _remove_unshaped_nodes(self, graph):
        for node_name, node_dict in list(graph.items()):
            if not node_dict["attr"]["input_shape"]:
                del graph[node_name]


class NNIBasedTorchConverter(NNIIRConverter):
    def __init__(self, model, example_inputs):
        torch = try_import_torch()
        try_import_nni()
        from nni.retiarii.converter import convert_to_graph
        from nni.retiarii.converter.graph_gen import GraphConverterWithShape

        # PyTorch module to NNI IR
        script_module = torch.jit.script(model)
        converter = GraphConverterWithShape()
        ir_model = convert_to_graph(
            script_module, model, converter=converter, dummy_input=example_inputs
        )

        super().__init__(ir_model)


class OnnxBasedTorchConverter(OnnxConverter):
    def __init__(self, model, example_inputs):
        onnx = try_import_onnx()
        torch = try_import_torch()
        with tempfile.TemporaryFile() as fp:
            torch.onnx.export(model, example_inputs, fp)
            fp.seek(0)
            model = onnx.load(fp, load_external_data=False)

        # convert model
        simplify = try_import_onnxsim()
        model_simp, check = simplify(model)

        assert check, "Simplified ONNX model could not be validated"
        super().__init__(model_simp)

