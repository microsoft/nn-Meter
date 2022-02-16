# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
logging = logging.getLogger("nn-Meter")


class ProtobufHelper:
    @staticmethod
    def get_w(x):
        """
        Get width from a list.

        Parameters
        ----------
        x : list
            A 2-D or 4-D list
                represent the shape of a tensor
        """
        l = len(x)
        if l == 4:
            return x[1]
        if l == 2:
            return x[0]
        return None

    @staticmethod
    def get_h(x):
        """
        Get height from a list.

        Parameters
        ----------
        x : list
            A 2-D or 4-D list
                represent the shape of a tensor
        """
        l = len(x)
        if l == 4:
            return x[2]
        if l == 2:
            return x[1]
        return None

    @staticmethod
    def find_weights_root(graph, node):
        """
        Find the node which store the weight of the tensor.

        Parameters
        ----------
        graph : dict
            The graph IR in dict form.
        node : dict
            A single node in graph IR.
        """
        NODE_WEIGHT_LUT = {
            "Conv2D": [
                lambda x: x.replace("/Conv2D", "/weight"),
                lambda x: x.replace("/Conv2D", "/kernel"),
            ],
            "DepthwiseConv2dNative": [lambda x: x.replace("/depthwise", "/weight")],
            "BiasAdd": [
                lambda x: x.replace("/BiasAdd", "/bias"),
            ],
            "FusedBatchNorm": [
                lambda x: x.replace("/FusedBatchNormV3", "/gamma"),
                lambda x: x.replace("/FusedBatchNormV3", "/beta"),
                lambda x: x.replace("/FusedBatchNormV3", "/moving_mean"),
                lambda x: x.replace("/FusedBatchNormV3", "/moving_variance"),
            ],
            "MatMul": [
                lambda x: x.replace("/MatMul", "/weight"),
            ],
        }

        weight_name = []
        if node["attr"]["type"] in NODE_WEIGHT_LUT.keys():
            for lut_lamba in NODE_WEIGHT_LUT[node["attr"]["type"]]:
                weight_op = lut_lamba(node["attr"]["name"])
                if (
                    weight_op in graph.keys()
                    and graph[weight_op]["attr"]["type"] != "Identity"
                ):
                    logging.info(
                        "Find node %s with its weight op %s."
                        % (node["attr"]["name"], weight_op)
                    )
                    weight_name.append(weight_op)

        return weight_name

    @staticmethod
    def get_graph_seq(graph, graph_head):
        """
        Run a topological sort of the graph,
        return the sorted sequence.

        Parameters
        ----------
        graph : dict
            The graph IR in dict form.
        graph_head : str
            Start position of the sort.
        """
        seen = set()
        stack = []
        order = []
        q = [graph_head[0]]
        for head in graph_head:
            q = [head]
            while q:
                v = q.pop()
                if v not in seen:
                    seen.add(v)
                    q.extend(graph[v]["outbounds"])
                    while stack and v not in graph[stack[-1]]["outbounds"]:
                        order.append(stack.pop())
                    stack.append(v)
        return stack + order[::-1]

    @staticmethod
    def pkg42dec(x):
        """
        Convert protobuf 4-packed oct format to number.

        Parameters
        ----------
        x : list
            The 4-packed oct list.
        """
        total_byte = len(x) // 4
        assert total_byte * 4 == len(x)

        num = []
        for idx in range(total_byte):
            num.append(0)
            for i in range(4):
                num[-1] += x[idx * 4 + i] << (i * 8)
            if num[-1] == 4294967295:
                num[-1] = -1

        return num

    @staticmethod
    def get_tensor_value(x):
        """
        Get the value from a const op.

        Parameters
        ----------
        x : Protobuf.node
            The const node.
        """
        DTYPE_ENUM = {
            0: lambda x: list(map(float, x.float_val)),
            1: lambda x: list(map(float, x.float_val)),
            3: lambda x: list(map(int, x.int_val)),
        }
        data = DTYPE_ENUM[x.dtype](x)
        if len(data) == 0:
            data = ProtobufHelper.pkg42dec(x.tensor_content)
        return data
