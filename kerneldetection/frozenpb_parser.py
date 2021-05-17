import tensorflow as tf
from google.protobuf import text_format
from tensorflow import gfile
from tensorflow import io

from shape_fetcher import ShapeFetcher

class FrozenPbParser:
    def __init__(self, pb_file):
        f = open(pb_file, 'rb')
        graph = tf.GraphDef()
        graph.ParseFromString(f.read())

        self.graph = graph

    def find_weights_root(self, node, shape_fetcher):
        if shape_fetcher == None:
            return None

        if node.op == 'Conv2D':
            weight_name = [node.name.replace('/Conv2D', '/weight/read'), node.name.replace('/Conv2D', '/kernel')]
        elif node.op == 'DepthwiseConv2dNative':
            weight_name = [node.name.replace('/depthwise', '/weight/read')]
        elif node.op == 'MatMul':
            weight_name = [node.name.replace('/MatMul',' /weight/read')]
        else:
            return None

        for target_node in self.graph.node:
            if target_node.name in weight_name:
                return [int(i) for i in shape_fetcher.shape_results[target_node.name + ':0']]

    def fetch_attr_to_dict(self, node, shape_fetcher):
        attr_dict = {}

        list_i_nodes = ['dilations', 'strides', 'ksize']
        str_nodes = ['padding', 'data_format']

        for attr_name in node.attr.keys():
            if attr_name in list_i_nodes:
                attr_dict[attr_name] = [int(a) for a in node.attr[attr_name].list.i]
                continue

            if attr_name in str_nodes:
                attr_dict[attr_name] = str(node.attr[attr_name].s)
                continue

            if attr_name == 'value':
                shape = []
                for dim in node.attr[attr_name].tensor.tensor_shape.dim:
                    shape.append(dim.size)
                attr_dict['tensor_shape'] = list(map(int, shape))
                continue

            if attr_name == 'shape':
                shape = []
                for dim in node.attr[attr_name].shape.dim:
                    shape.append(dim.size)
                attr_dict['shape'] = list(map(int, shape))
                continue

        attr_dict['weight_shape'] = self.find_weights_root(node, shape_fetcher)


        return attr_dict


    def parse_graph(self, graph_helper, required_shape=False, insert_node=False):
        if required_shape:
            shape_fetcher = ShapeFetcher(self.graph)

        for node in self.graph.node:
            graph_helper.node(str(node.name), list(map(str,node.input)))
            graph_helper.set_node_attr(node.name, {
                'type': str(node.op),
                'output_shape': [int(i) for i in shape_fetcher.shape_results[node.name + ':0']] if required_shape else [],
                'attr': self.fetch_attr_to_dict(node, shape_fetcher if required_shape else None),
                'node': node if insert_node else None
            })