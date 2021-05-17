class MatchHelper:
    base_type_table = {
        'ReLU': [
            'Relu',
            'Relu6',
            'ReLU',
            'ReLU6',
        ],
        'BatchNorm': [
            'BatchNorm',
            'FusedBatchNorm',
            'FusedBatchNormV2',
            'FusedBatchNormV3',
        ],
        'TwoInputElementWise': [
            'BiasAdd',
            'Add',
            'Mul',
        ],
        'DepthwiseConv2D': [
            'DepthwiseConv2dNative',
        ],
        'FC': [
            'MatMul',
        ]
    }

    @classmethod
    def get_base_type(cls, node_type):
        for key, value in cls.base_type_table.items():
            if node_type in value:
                return key
        return node_type

    @classmethod
    def op_type_matcher(cls, node_1, node_2):
        def get_ast_by_op(op_name):
            for key, value in cls.base_type_table.items():
                if op_name in value:
                    return key

            return op_name

        if 'type' in node_1 and 'type' in node_2:
            if '_tagged' in node_1 or '_tagged' in node_2:
                return False

            if node_1['type'] == 'dummy' or node_2['type'] == 'dummy':
                return True
            return get_ast_by_op(node_1['type']) == get_ast_by_op(node_2['type'])
        else:
            return False

    @staticmethod
    def strip_useless_nodes(graph_helper):
        stripped_nodes = ['Const', 'Identity']

        graph = graph_helper.get_graph()
        removed_node = []
        for key, value in graph.items():
            if value['attr']['type'] in stripped_nodes:
                removed_node.append(key)

        for key in removed_node:
            del graph[key]

        graph_helper.refresh()

    @staticmethod
    def tag_matched_nodes(grapher, matched_subgraph):
        for matched_unit in matched_subgraph:
            for node_name in matched_unit.keys():
                grapher.add_node_attr(node_name, '_tagged', '')

    @staticmethod
    def get_untagged_nodes(grapher):
        untagged_node = []
        for node in grapher.get_graph().keys():
            if '_tagged' not in grapher.get_node_attr(node)['attr']:
                untagged_node.append(node)
        return untagged_node