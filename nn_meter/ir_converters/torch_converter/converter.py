import torch
import onnx
import tempfile
from nn_meter.ir_converters.onnx_converter import OnnxConverter

from nni.retiarii.converter import convert_to_graph
from nni.retiarii.converter.graph_gen import GraphConverterWithShape
from nni.retiarii.graph import Model

from .opset_map import nni_attr_map, nni_type_map


class NNIIRConverter:
    def __init__(self, ir_model: Model):
        self.ir_model = ir_model.fork()
        GraphConverterWithShape().flatten(self.ir_model)

    def convert(self):
        graphe = self._to_graphe_layout()

        for _, node in graphe.items():
            self._map_opset(node)

        self._remove_unshaped_nodes(graphe)

        return graphe

    def _to_graphe_layout(self):
        graphe = {}

        for node in self.ir_model.root_graph.hidden_nodes:
            node_dict = {
                'attr': {
                    'attr': {
                        k: v
                        for k, v in node.operation.parameters.items()
                        if k not in ['input_shape', 'output_shape']
                    },
                    'input_shape': node.operation.parameters['input_shape'],
                    'output_shape': node.operation.parameters['output_shape'],
                    'type': node.operation.type,
                },
                'inbounds': [],
                'outbounds': [],
            }

            incoming_edges = sorted(node.incoming_edges, key=lambda e: e.tail_slot or 0)
            for edge in incoming_edges:
                node_dict['inbounds'].append(edge.head.name)

            outgoing_edges = sorted(node.outgoing_edges, key=lambda e: e.head_slot or 0)
            for edge in outgoing_edges:
                node_dict['outbounds'].append(edge.tail.name)

            graphe[node.name] = node_dict

        return graphe

    def _map_opset(self, node):
        old_type = node['attr']['type']
        new_type = nni_type_map.get(old_type, old_type)

        new_attr_dict = {}
        for attr_name, attr_value in node['attr']['attr'].items():
            new_attr_name = attr_name
            new_attr_value = attr_value
            for type, attr_map in nni_attr_map.items():
                if type == '__all__' or type == new_type:
                    if attr_name in attr_map:
                        new_attr_name, modifier = attr_map[attr_name]
                        if modifier is not None:
                            new_attr_value = modifier(attr_value)

            new_attr_dict[new_attr_name] = new_attr_value

        node['attr']['type'] = new_type
        node['attr']['attr'] = new_attr_dict

    def _remove_unshaped_nodes(self, graphe):
        for node_name, node_dict in list(graphe.items()):
            if not node_dict['attr']['input_shape']:
                del graphe[node_name]


class NNIBasedTorchConverter(NNIIRConverter):
    def __init__(self, model, example_inputs):
        # PyTorch module to NNI IR
        script_module = torch.jit.script(model)
        converter = GraphConverterWithShape()
        ir_model = convert_to_graph(script_module, model, converter, example_inputs=example_inputs)

        super().__init__(ir_model)


class OnnxBasedTorchConverter(OnnxConverter):
    def __init__(self, model, example_inputs):
        with tempfile.TemporaryFile() as fp:
            torch.onnx.export(model, example_inputs, fp)
            fp.seek(0)
            model = onnx.load(fp, load_external_data=False)

        super().__init__(model)


TorchConverter = NNIBasedTorchConverter
