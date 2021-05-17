from .protobuf_helper import ProtobufHelper as ph
from functools import reduce
import copy
import logging
logging = logging.getLogger(__name__)


class ShapeInference:

    @staticmethod
    def eval_prodcast(grapher, node):
        input_nodes = node['inbounds']
        if len(input_nodes) < 2:
            logging.warn(
                'Invalid input op num for prodcast op %s' %
                (node['name']))
            if len(input_nodes) == 1:
                return grapher[node['inbounds'][0]]['attr']['output_shape'][0]
            else:
                return None

        target_dim = -1
        target_shape = [1]
        input_shape_list = []
        for node_name in input_nodes:
            input_shape = grapher[node_name]['attr']['output_shape'][0]
            input_shape_list.append(input_shape)
            if target_dim < len(input_shape):
                target_dim = len(input_shape)
                target_shape = input_shape
            elif target_dim == len(input_shape):
                for i in range(target_dim):
                    if target_shape[i] < input_shape[i]:
                        target_shape[i] = input_shape[i]

        #     if target_dim < len(input_shape):
        #         for i in range(len(input_shape)):
        #             if target_shape[i] == 1 or target_shape[i] == input_shape[i]:
        #                 target_dim = len(input_shape)
        #                 target_shape = input_shape
        #             else:
        #                 logging.warn('Invalid prodcast shape between %s and %s(%s).'
        #                     % (str(target_shape), str(input_shape), node_name))
        #                 return None

        #         logging.warn('Prodcast from %s to %s(%s).' % (str(target_shape), str(input_shape), node_name))

        # for node_name in input_nodes:
        #     input_shape = grapher[node_name]['attr']['output_shape'][0]
        #     if largest_dim == len(input_shape):
        #         prodcast_shape[node_name] = input_shape

        # for node_name, shape in prodcast_shape.items():
        #     if shape != prodcast_shape[prodcast_node_name]:
        #         logging.warn('Invalid prodcast shape between %s(%s) and %s(%s).'
        #             % (node_name, str(shape),
        #                 prodcast_node_name, prodcast_shape[prodcast_node_name]))
        #         return None

        return input_shape_list, [target_shape]

    @staticmethod
    def Const_get_shape(grapher, node):
        return [], [node['attr']['attr']['tensor_shape']]

    @staticmethod
    def Identity_get_shape(grapher, node):
        return [], [grapher[node['inbounds'][0]]['attr']['output_shape'][0]]

    @staticmethod
    def propogate_shape(grapher, node):
        in_shape = [grapher[node['inbounds'][0]]['attr']['output_shape'][0]]
        return in_shape, in_shape

    @staticmethod
    def FusedBatchNorm_get_shape(grapher, node):
        return ShapeInference.propogate_shape(grapher, node)

    @staticmethod
    def BiasAdd_get_shape(grapher, node):
        return ShapeInference.propogate_shape(grapher, node)

    @staticmethod
    def Relu_get_shape(grapher, node):
        return ShapeInference.propogate_shape(grapher, node)

    @staticmethod
    def Relu6_get_shape(grapher, node):
        return ShapeInference.propogate_shape(grapher, node)

    @staticmethod
    def LeakyReLU_get_shape(grapher, node):
        return ShapeInference.propogate_shape(grapher, node)

    @staticmethod
    def Add_get_shape(grapher, node):
        return ShapeInference.eval_prodcast(grapher, node)

    @staticmethod
    def Mul_get_shape(grapher, node):
        return ShapeInference.eval_prodcast(grapher, node)

    @staticmethod
    def Pool_get_shape(grapher, node):
        if len(node['inbounds']) != 1:
            logging.warning(
                'Failed to get input node of %s.' %
                (node['attr']['name']))
            logging.info(node)
            return

        input_shape = grapher[node['inbounds'][0]]['attr']['output_shape'][0]
        logging.info(
            'Get input shape of %s from %s, input shape:%s.' %
            (node['attr']['name'], node['inbounds'][0], input_shape))

        k_size = node['attr']['attr']['ksize']

        if node['attr']['attr']['strides'][::3] != [1, 1]:
            logging.warning(
                'Invalid strides %s of node %s.' %
                (str(
                    node['attr']['attr']['strides']),
                    node['attr']['name']))
            logging.info(node)
            return

        strides = node['attr']['attr']['strides']
        padding = node['attr']['attr']['padding'].decode('utf-8')
        logging.info('Op:%s, stride:%s, padding:%s.' %
                     (node['attr']['name'], str(strides), str(padding)))

        if padding == 'SAME':
            wpad = ph.get_w(strides) - 1
            hpad = ph.get_h(strides) - 1
        else:
            wpad = 0
            hpad = 0
        padded_shape = [
            ph.get_w(input_shape) + wpad,
            ph.get_h(input_shape) + hpad]
        logging.info('Op:%s, padding:%s, padded shape:%s.' %
                     (node['attr']['name'], str([wpad, hpad]), str(padded_shape)))

        outw = int(ph.get_w(input_shape) - ph.get_w(k_size)) / \
            ph.get_w(strides) + 1
        outh = int(ph.get_h(input_shape) - ph.get_w(k_size)) / \
            ph.get_h(strides) + 1

        output_shape = list(
            map(int, [input_shape[0], outh, outw, input_shape[3]]))
        return [input_shape], [output_shape]

    @staticmethod
    def AvgPool_get_shape(grapher, node):
        return ShapeInference.Pool_get_shape(grapher, node)

    @staticmethod
    def AveragePooling2D_get_shape(grapher, node):
        return ShapeInference.Pool_get_shape(grapher, node)

    @staticmethod
    def MaxPool_get_shape(grapher, node):
        return ShapeInference.Pool_get_shape(grapher, node)

    @staticmethod
    def MaxPooling2D_get_shape(grapher, node):
        return ShapeInference.Pool_get_shape(grapher, node)

    @staticmethod
    def Placeholder_get_shape(grapher, node):
        return [], [node['attr']['attr']['shape']]

    @staticmethod
    def Conv2D_get_shape(grapher, node):
        weight_node = ph.find_weights_root(grapher, node)
        if len(weight_node) != 1:
            logging.warning(
                'Failed to get shape of node %s.' %
                (node['attr']['name']))
            logging.info(node)
            return

        input_node = [x for x in node['inbounds'] if x != weight_node]
        input_node = [x for x in input_node if grapher[x]
                      ['attr']['type'] != 'Identity']
        if len(input_node) != 1:
            logging.warning(
                'Failed to get input node of %s.' %
                (node['attr']['name']))
            logging.info(node)
            return

        input_shape = grapher[input_node[0]]['attr']['output_shape'][0]
        logging.info(
            'Get input shape of %s from %s, input shape:%s.' %
            (node['attr']['name'], input_node[0], input_shape))

        weight_shape = grapher[weight_node[0]]['attr']['attr']['tensor_shape']
        if len(weight_shape) != 4:
            logging.warning(
                'Failed to parse weight shape %s of node %s.' %
                (str(weight_shape), node['attr']['name']))
            logging.info(node)
            return

        logging.info(
            'Get weight shape of %s from %s, input shape:%s.' %
            (node['attr']['name'], weight_node, weight_shape))

        k_size = weight_shape[:2]
        cin = weight_shape[2]
        cout = weight_shape[3]

        if node['attr']['attr']['strides'][::3] != [1, 1]:
            logging.warning(
                'Invalid strides %s of node %s.' %
                (str(
                    node['attr']['attr']['strides']),
                    node['attr']['name']))
            logging.info(node)
            return

        strides = node['attr']['attr']['strides']
        dilation = node['attr']['attr']['dilations']
        padding = node['attr']['attr']['padding'].decode('utf-8')
        logging.info(
            'Op:%s, stride:%s, dilation:%s, padding:%s.' %
            (node['attr']['name'], str(strides), str(dilation), str(padding)))

        kernel_extent_w = ph.get_w(dilation) * (ph.get_w(strides) - 1) + 1
        kernel_extent_h = ph.get_h(dilation) * (ph.get_h(strides) - 1) + 1

        if padding == 'SAME':
            wpad = kernel_extent_w + int((ph.get_w(input_shape) - 1) / ph.get_w(
                dilation)) * ph.get_w(dilation) - ph.get_w(input_shape)
            hpad = kernel_extent_h + int((ph.get_h(input_shape) - 1) / ph.get_h(
                dilation)) * ph.get_h(dilation) - ph.get_h(input_shape)
        else:
            wpad = 0
            hpad = 0
        padded_shape = [
            ph.get_w(input_shape) + wpad,
            ph.get_h(input_shape) + hpad]
        logging.info('Op:%s, kernel_extent:%s, padding:%s, padded shape:%s.' % (node['attr']['name'], str(
            [kernel_extent_w, kernel_extent_h]), str([wpad, hpad]), str(padded_shape)))

        outw = int(ph.get_w(input_shape) - kernel_extent_w) / \
            ph.get_w(strides) + 1
        outh = int(ph.get_h(input_shape) - kernel_extent_h) / \
            ph.get_h(strides) + 1

        output_shape = list(map(int, [input_shape[0], outh, outw, cout]))
        return [input_shape], [output_shape]

    @staticmethod
    def DepthwiseConv2dNative_get_shape(grapher, node):
        weight_node = ph.find_weights_root(grapher, node)
        if len(weight_node) != 1:
            logging.warning(
                'Failed to get shape of node %s.' %
                (node['attr']['name']))
            logging.info(node)
            return

        input_node = [x for x in node['inbounds'] if x != weight_node]
        input_node = [x for x in input_node if grapher[x]
                      ['attr']['type'] != 'Identity']
        if len(input_node) != 1:
            logging.warning(
                'Failed to get input node of %s.' %
                (node['attr']['name']))
            logging.info(node)
            return

        input_shape = grapher[input_node[0]]['attr']['output_shape'][0]
        logging.info(
            'Get input shape of %s from %s, input shape:%s.' %
            (node['attr']['name'], input_node[0], input_shape))

        weight_shape = grapher[weight_node[0]]['attr']['attr']['tensor_shape']
        if len(weight_shape) != 4:
            logging.warning(
                'Failed to parse weight shape %s of node %s.' %
                (str(weight_shape), node['attr']['name']))
            logging.info(node)
            return

        logging.info(
            'Get weight shape of %s from %s, input shape:%s.' %
            (node['attr']['name'], weight_node, weight_shape))

        k_size = weight_shape[:2]
        cin = weight_shape[2]

        if node['attr']['attr']['strides'][::3] != [1, 1]:
            logging.warning(
                'Invalid strides %s of node %s.' %
                (str(
                    node['attr']['attr']['strides']),
                    node['attr']['name']))
            logging.info(node)
            return

        strides = node['attr']['attr']['strides']
        dilation = node['attr']['attr']['dilations']
        padding = node['attr']['attr']['padding'].decode('utf-8')
        logging.info(
            'Op:%s, stride:%s, dilation:%s, padding:%s.' %
            (node['attr']['name'], str(strides), str(dilation), str(padding)))

        kernel_extent_w = ph.get_w(dilation) * (ph.get_w(strides) - 1) + 1
        kernel_extent_h = ph.get_h(dilation) * (ph.get_h(strides) - 1) + 1

        if padding == 'SAME':
            wpad = kernel_extent_w + int((ph.get_w(input_shape) - 1) / ph.get_w(
                dilation)) * ph.get_w(dilation) - ph.get_w(input_shape)
            hpad = kernel_extent_h + int((ph.get_h(input_shape) - 1) / ph.get_h(
                dilation)) * ph.get_h(dilation) - ph.get_h(input_shape)
        else:
            wpad = 0
            hpad = 0
        padded_shape = [
            ph.get_w(input_shape) + wpad,
            ph.get_h(input_shape) + hpad]
        logging.info('Op:%s, kernel_extent:%s, padding:%s, padded shape:%s.' % (node['attr']['name'], str(
            [kernel_extent_w, kernel_extent_h]), str([wpad, hpad]), str(padded_shape)))

        outw = int(ph.get_w(input_shape) - kernel_extent_w) / \
            ph.get_w(strides) + 1
        outh = int(ph.get_h(input_shape) - kernel_extent_h) / \
            ph.get_h(strides) + 1

        output_shape = list(map(int, [input_shape[0], outh, outw, cin]))
        return [input_shape], [output_shape]

    @staticmethod
    def Reduce_get_shape(grapher, node):
        input_shape = grapher[node['inbounds'][0]]['attr']['output_shape'][0]
        output_shape = input_shape
        logging.info(
            'Get input shape of %s from %s, input shape:%s.' %
            (node['attr']['name'], node['inbounds'][0], output_shape))

        output_shape[1] = 0
        output_shape[2] = 0

        reduction_indices = node['attr']['attr']['reduction_indices']
        logging.info('Get Reduction Indices %s.', str(reduction_indices))

        reduction_cnt = 0
        for reduction in sorted(reduction_indices):
            del output_shape[reduction - reduction_cnt]
            reduction_cnt += 1

        return [input_shape], [output_shape]

    @staticmethod
    def Mean_get_shape(grapher, node):
        return ShapeInference.Reduce_get_shape(grapher, node)

    @staticmethod
    def GlobalAveragePooling2D_get_shape(grapher, node):
        return ShapeInference.Reduce_get_shape(grapher, node)

    @staticmethod
    def GlobalMaxPooling2D_get_shape(grapher, node):
        return ShapeInference.Reduce_get_shape(grapher, node)

    @staticmethod
    def MatMul_get_shape(grapher, node):
        weight_node = ph.find_weights_root(grapher, node)
        if len(weight_node) != 1:
            logging.warning(
                'Failed to get shape of node %s.' %
                (node['attr']['name']))
            logging.info(node)
            return

        weight_shape = grapher[weight_node[0]]['attr']['attr']['tensor_shape']
        if len(weight_shape) != 2:
            logging.warning(
                'Failed to parse weight shape %s of node %s.' %
                (str(weight_shape), node['attr']['name']))
            logging.info(node)
            return

        logging.info(
            'Get weight shape of %s from %s, input shape:%s.' %
            (node['attr']['name'], weight_node, weight_shape))

        input_node = [x for x in node['inbounds'] if x != weight_node]
        input_node = [x for x in input_node if grapher[x]
                      ['attr']['type'] != 'Identity']
        if len(input_node) != 1:
            logging.warning(
                'Failed to get input node of %s.' %
                (node['attr']['name']))
            logging.info(node)
            return

        input_shape = copy.deepcopy(
            grapher[input_node[0]]['attr']['output_shape'][0])
        logging.info(
            'Get input shape of %s from %s, input shape:%s.' %
            (node['attr']['name'], input_node[0], input_shape))

        if weight_shape[0] != input_shape[1]:
            logging.warning(
                'Weight shape and input shape not matched for %s.' %
                (node['attr']['name']))
            logging.info(node)
            return

        output_shape = copy.deepcopy(input_shape)
        output_shape[1] = weight_shape[1]

        return [input_shape], [output_shape]

    @staticmethod
    def Reshape_get_shape(grapher, node):
        input_shape = grapher[node['inbounds'][0]]['attr']['output_shape'][0]
        exp_output_shape = node['attr']['attr']['shape']

        input_elements = abs(reduce(lambda x, y: x * y, input_shape))
        exp_output_shape_elements = abs(
            reduce(lambda x, y: x * y, exp_output_shape))

        if input_elements != exp_output_shape_elements:
            logging.warning('Input shape %s and output shape %s not matched for %s.' % (
                str(input_shape, str(output_shape), node['attr']['name'])))

        return [input_shape], [exp_output_shape]

    @staticmethod
    def Concat_get_shape(grapher, node):
        input_shape = []
        for in_node in node['inbounds']:
            in_shape = grapher[in_node]['attr']['output_shape'][0]
            if in_shape != []:
                input_shape.append(in_shape)
                logging.info('Get input shape of %s from %s, input shape:%s.' % (
                    node['attr']['name'], in_node, input_shape[-1]))

        axis = node['attr']['attr']['axis'][0]

        output_shape = copy.deepcopy(input_shape[0])
        for in_shape in input_shape[1:]:
            output_shape[axis] += in_shape[axis]

        return [input_shape], [output_shape]

    @staticmethod
    def Concatenate_get_shape(grapher, node):
        return ShapeInference.Concat_get_shape(grapher, node)

    @staticmethod
    def ConcatV2_get_shape(grapher, node):
        return ShapeInference.Concat_get_shape(grapher, node)

    @staticmethod
    def Split_get_shape(grapher, node):
        raise NotImplementedError

    @staticmethod
    def StridedSlice_get_shape(grapher, node):
        return None, None

    @staticmethod
    def Pack_get_shape(grapher, node):
        return None, None

    def __init__(self, grapher):
        seq = ph.get_graph_seq(grapher)
        graph = grapher.get_graph()
        for node_name in seq:
            node_get_shape_name = grapher.get_node_type(
                node_name) + '_get_shape'
            if node_get_shape_name in dir(self):
                input_shape, output_shape = eval(
                    'self.' + node_get_shape_name)(graph, graph[node_name])
                if output_shape is not None:
                    graph[node_name]['attr']['output_shape'] = output_shape
                if input_shape is not None:
                    graph[node_name]['attr']['input_shape'] = input_shape
                logging.info(
                    'Input shape of %s op is %s.' %
                    (node_name, str(input_shape)))
                logging.info(
                    'Output shape of %s op is %s.' %
                    (node_name, str(output_shape)))
            else:
                logging.warning(
                    'Op %s is not support, ignored!' %
                    grapher.get_node_type(node_name))
