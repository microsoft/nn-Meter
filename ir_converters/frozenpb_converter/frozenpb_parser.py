from .protobuf_helper import ProtobufHelper
from .shape_fetcher import ShapeFetcher
from tensorflow import io
from tensorflow import gfile
from google.protobuf import text_format
import tensorflow as tf
import copy
import re
import logging
logging = logging.getLogger(__name__)


class FrozenPbParser:
    def __init__(self, pb_file):
        f = open(pb_file, 'rb')
        graph = tf.GraphDef()
        graph.ParseFromString(f.read())

        self.graph = graph

    @staticmethod
    def strip_useless_nodes(graph_helper):
        stripped_nodes_type = ['Const', 'Identity']
        stripped_nodes_keywords = ['/weight', '/weight/read',
                                   '/ReadVariableOp',
                                   '/kernel', '/gamma',
                                   '/beta', '/moving_mean',
                                   '/moving_variance',
                                   '/bias', '/reduction_indices',
                                   '/shape', '/split_dim', '/axis']
        graph = graph_helper.get_graph()
        removed_node = []
        for key, value in graph.items():
            if 'attr' in value.keys():
                if value['attr']['type'] in stripped_nodes_type:
                    for kw in stripped_nodes_keywords:
                        if kw in key:
                            removed_node.append(key)
                            break

        for key in removed_node:
            del graph[key]

        graph_helper.refresh()
        graph_helper.refresh()
        graph_helper.refresh()
        graph_helper.refresh()
        graph_helper.refresh()

    @staticmethod
    def fix_split_naming(graph_helper):
        graph = graph_helper.get_graph()
        graph_nodes = copy.deepcopy(list(graph.keys()))
        remove_node_list = []
        for graph_node in graph_nodes:
            if graph_node in graph.keys():
                if 'attr' in graph[graph_node].keys():
                    if graph[graph_node]['attr']['type'] == 'Split' and ':' not in graph_node:
                        logging.info('Find split main node %s.' % graph_node)
                        split_node_name = graph_node
                        split_node_child = []
                        for node_name in graph.keys():
                            idx = re.findall(
                                r'%s:(\d+)' %
                                split_node_name, node_name)
                            if len(idx) > 0:
                                idx = int(idx[0])
                                logging.info(
                                    'Find split child node %s.' % node_name)
                                graph[graph_node]['outbounds'] += graph[node_name]['outbounds']
                                graph[graph[node_name]['outbounds']
                                      [0]]['inbounds'] += [graph_node]
                                remove_node_list.append(node_name)

        for node in remove_node_list:
            del graph[node]

        graph_helper.refresh()
        graph_helper.refresh()
        graph_helper.refresh()
        graph_helper.refresh()
        graph_helper.refresh()

    def fetch_attr_to_dict(self, node, shape_fetcher):
        attr_dict = {}

        attr_as_node = {
            'Split': {
                'node_name': lambda x: x + '/split_dim',
                'attr_name': 'split_dim',
                'node_value': lambda x: ProtobufHelper.get_tensor_value(x)
            },
            'Mean': {
                'node_name': lambda x: x + '/reduction_indices',
                'attr_name': 'reduction_indices',
                'node_value': lambda x: ProtobufHelper.pkg42dec(x.tensor_content)
            },
            'Reshape': {
                'node_name': lambda x: x + '/shape',
                'attr_name': 'shape',
                'node_value': lambda x: ProtobufHelper.pkg42dec(x.tensor_content)
            },
            'Concat': {
                'node_name': lambda x: x + '/axis',
                'attr_name': 'axis',
                'node_value': lambda x: ProtobufHelper.get_tensor_value(x)
            },
            'ConcatV2': {
                'node_name': lambda x: x + '/axis',
                'attr_name': 'axis',
                'node_value': lambda x: ProtobufHelper.get_tensor_value(x)
            },
            'Const': {
                'node_name': lambda x: x,
                'attr_name': 'constant',
                'node_value': lambda x: ProtobufHelper.get_tensor_value(x)
            }
        }

        list_i_nodes = ['dilations', 'strides', 'ksize']
        str_nodes = ['padding', 'data_format']

        for attr_name in node.attr.keys():
            if attr_name in list_i_nodes:
                attr_dict[attr_name] = [
                    int(a) for a in node.attr[attr_name].list.i]
                continue

            if attr_name in str_nodes:
                attr_dict[attr_name] = node.attr[attr_name].s
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

        if node.op in attr_as_node.keys():
            for target_node in self.graph.node:
                if target_node.name == attr_as_node[node.op]['node_name'](
                        node.name):
                    for attr_name in target_node.attr.keys():
                        if attr_name == 'value' and 'weight' not in node.name and 'BatchNorm' not in node.name and 'kernel' not in node.name:
                            # print(target_node.attr[attr_name].tensor)
                            attr_dict[attr_as_node[node.op]['attr_name']] = \
                                attr_as_node[node.op]['node_value'](target_node.attr[attr_name].tensor)

        # # attr_dict['weight_shape'] = self.find_weights_root(node, shape_fetcher)
        # print(node.name, attr_dict)
        # print('------------------')
        return attr_dict

    def parse_graph(
            self,
            graph_helper,
            required_shape=False,
            insert_node=False):
        if required_shape:
            shape_fetcher = ShapeFetcher(self.graph)

        for node in self.graph.node:
            graph_helper.node(str(node.name), list(map(str, node.input)))
            graph_helper.set_node_attr(
                node.name, {
                    'name': str(node.name),
                    'type': str(node.op),
                    'output_shape': shape_fetcher.shape_results[node.name + ':0'] if required_shape else [],
                    'attr': self.fetch_attr_to_dict(node, shape_fetcher if required_shape else None),
                    # 'node': node if insert_node else None
                })

        # return shape_fetcher