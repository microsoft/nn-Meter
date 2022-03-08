# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from functools import reduce
import copy
import math
import logging
from .protobuf_helper import ProtobufHelper as ph
logging = logging.getLogger("nn-Meter")


class ShapeInference:

    # Ops that only need to calculate the prodcast for shapes
    TF_PRODCAST_MATH_OPS = [
        "Add",
        "AddN", # AddN does not support prodcast really
        "AddV2",
        "Subtract",
        "Sub",
        "MulNoNan",
        "Multiply",
        "Mul",
        "Div",
        "DivNoNan",
        "Equal",
    ]

    # Ops that only need to propagate the shapes
    TF_PROPAGATE_MATH_OPS = [
        "Abs",
        "Acos",
        "ACosh",
        "Asin",
        "Asinh",
        "Atan",
        "Atanh",
        "Cos",
        "Cosh",
        "Exp",
        "Floor",
        "Sin",
        "Sinh",
        "Sqrt",
        "Square",

        "FusedBatchNorm",
        "FusedBatchNormV2",
        "FusedBatchNormV3",
        "BiasAdd",

        "Relu",
        "Relu6",
        "Selu",
        "LeakyReLU",
        "Elu"
        "Softmax",

        "NoOp"
    ]

    @staticmethod
    def eval_prodcast(graph, node):
        """
        Evalute the prodcast along the input nodes.
        Now we use lazy prodcast, will move to tf implement later.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        input_nodes = node["inbounds"]
        if len(input_nodes) < 2:
            logging.warn(
                "Invalid input op num for prodcast op %s" % (node["attr"]["name"])
            )
            if len(input_nodes) == 1:
                return graph[node["inbounds"][0]]["attr"]["output_shape"][0]
            else:
                return None

        target_dim = -1
        target_shape = [1]
        input_shape_list = []
        for node_name in input_nodes:
            input_shape = graph[node_name]["attr"]["output_shape"][0]
            input_shape_list.append(input_shape)
            if target_dim < len(input_shape):
                target_dim = len(input_shape)
                target_shape = input_shape
            elif target_dim == len(input_shape):
                for i in range(target_dim):
                    if target_shape[i] < input_shape[i]:
                        target_shape[i] = input_shape[i]

        return input_shape_list, [target_shape]

    @staticmethod
    def get_padding_shape(input_shape, cout, k_size, strides, padding):
        """
        Calculate the padded shape of a given tensor.

        Parameters
        ----------
        input_shape : list
            Input shape in list, a 2D or 4D list.
        cout : int
            Cout of the operation.
        k_size : list
            Kernel size of the operation, a 2D or 4D list.
        strides : list
            Strides of the operation, a 2D or 4D list.
        padding : str
            Padding type, now support SAME and VALID.
        """

        logging.info(
            "Calculating padding shape, input shape: %s, kernel size: %s, strides: %s, padding: %s."
            % (str(input_shape), str(k_size), str(strides), str(padding))
        )

        if padding == "SAME":
            outh = math.ceil(ph.get_h(input_shape) / ph.get_h(strides))
            outw = math.ceil(ph.get_w(input_shape) / ph.get_w(strides))

            padh = max(
                (outh - 1) * ph.get_h(strides)
                + ph.get_h(k_size)
                - ph.get_h(input_shape),
                0,
            )
            padw = max(
                (outw - 1) * ph.get_w(strides)
                + ph.get_w(k_size)
                - ph.get_w(input_shape),
                0,
            )

            pad_top = padh // 2
            pad_bottom = padh - pad_top
            pad_left = padw // 2
            pad_right = padw - pad_left

            pad_size = [pad_top, pad_bottom, pad_left, pad_right]
        elif padding == "VALID":
            outh = math.ceil(
                (ph.get_h(input_shape) - ph.get_h(k_size) + 1) / ph.get_h(strides)
            )
            outw = math.ceil(
                (ph.get_h(input_shape) - ph.get_h(k_size) + 1) / ph.get_w(strides)
            )

            pad_size = [0, 0, 0, 0]
        else:
            logging.error("Unexpected padding format %s." % (padding))
            return None, None

        output_shape = list(map(int, [input_shape[0], outh, outw, cout]))
        return copy.deepcopy(output_shape), copy.deepcopy(pad_size)

    @staticmethod
    def Const_get_shape(graph, node):
        """
        Get shape of a const operator.
        Take the tensor shape as the output shape.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        return [], [node["attr"]["attr"]["tensor_shape"]]

    @staticmethod
    def Identity_get_shape(graph, node):
        """
        Get shape of an Identity operator.
        This is not well implemented, will move a more robust later.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        return [], [graph[node["inbounds"][0]]["attr"]["output_shape"][0]]
    
    @staticmethod
    def Pad_get_shape(graph, node):
        """
        Get shape of a Pad operator.
        This is not well implemented, will move a more robust later.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        in_shape = [graph[node["inbounds"][0]]["attr"]["output_shape"][0]]
        paddings = node["attr"]["attr"]["paddings"]
        out_shape = []
        for dim in range(len(in_shape)):
            out_shape.append(in_shape[dim] + sum(paddings[dim]))
        return in_shape, out_shape
    
    @staticmethod
    def PadV2_get_shape(graph, node):
        """
        Get shape of a PadV2 operator.
        This is not well implemented, will move a more robust later.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        return ShapeInference.Pad_get_shape(graph, node)

    @staticmethod
    def propagate_shape(graph, node):
        """
        For operator who does not affect the shapes.
        Just take the input shapes as output.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        logging.info("Propagate through op %s.", node["attr"]["name"])
        in_shape = [graph[node["inbounds"][0]]["attr"]["output_shape"][0]]
        return in_shape, in_shape

    @staticmethod
    def Pool_get_shape(graph, node):
        """
        Get shape of a Pooling type operation.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        if len(node["inbounds"]) != 1:
            logging.warning("Failed to get input node of %s." % (node["attr"]["name"]))
            logging.info(node)
            return

        input_shape = copy.deepcopy(
            graph[node["inbounds"][0]]["attr"]["output_shape"][0]
        )
        logging.info(
            "Get input shape of %s from %s, input shape:%s."
            % (node["attr"]["name"], node["inbounds"][0], input_shape)
        )

        k_size = node["attr"]["attr"]["ksize"]

        if node["attr"]["attr"]["strides"][::3] != [1, 1]:
            logging.warning(
                "Invalid strides %s of node %s."
                % (str(node["attr"]["attr"]["strides"]), node["attr"]["name"])
            )
            logging.info(node)
            return

        strides = node["attr"]["attr"]["strides"]
        padding = node["attr"]["attr"]["padding"].decode("utf-8")
        logging.info(
            "Op:%s, stride:%s, padding:%s."
            % (node["attr"]["name"], str(strides), str(padding))
        )

        out_shape, padding_shape = ShapeInference.get_padding_shape(
            input_shape, input_shape[3], k_size, strides, padding
        )

        node["attr"]["attr"]["ksize"] = copy.deepcopy(
            node["attr"]["attr"]["ksize"][1:-1]
        )
        node["attr"]["attr"]["strides"] = copy.deepcopy(
            node["attr"]["attr"]["strides"][1:-1]
        )
        node["attr"]["attr"]["pads"] = copy.deepcopy(padding_shape)

        return [input_shape], [out_shape]

    @staticmethod
    def AvgPool_get_shape(graph, node):
        """
        Get shape of an AvgPool operator.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        return ShapeInference.Pool_get_shape(graph, node)

    @staticmethod
    def AveragePooling2D_get_shape(graph, node):
        """
        Get shape of an AveragePooling2D operator.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        return ShapeInference.Pool_get_shape(graph, node)

    @staticmethod
    def MaxPool_get_shape(graph, node):
        """
        Get shape of a MaxPool operator.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        return ShapeInference.Pool_get_shape(graph, node)
    
    @staticmethod
    def MaxPoolV2_get_shape(graph, node):
        """
        Get shape of a MaxPool operator.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        return ShapeInference.Pool_get_shape(graph, node)

    @staticmethod
    def MaxPooling2D_get_shape(graph, node):
        """
        Get shape of a MaxPooling2D operator.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        return ShapeInference.Pool_get_shape(graph, node)

    @staticmethod
    def Placeholder_get_shape(graph, node):
        """
        Get shape of a Placeholder operator.
        This fetch the shape from the shape attribute of the operator.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        return [], [node["attr"]["attr"]["shape"]]

    @staticmethod
    def Conv2D_get_shape(graph, node):
        """
        Get shape of a Conv2D operator.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        weight_node = ph.find_weights_root(graph, node)
        if len(weight_node) != 1:
            logging.warning("Failed to get shape of node %s." % (node["attr"]["name"]))
            logging.info(node)
            return

        input_node = [x for x in node["inbounds"] if x != weight_node]
        input_node = [x for x in input_node if graph[x]["attr"]["type"] != "Identity"]
        if len(input_node) != 1:
            logging.warning("Failed to get input node of %s." % (node["attr"]["name"]))
            logging.info(node)
            return

        input_shape = copy.deepcopy(graph[input_node[0]]["attr"]["output_shape"][0])
        logging.info(
            "Get input shape of %s from %s, input shape:%s."
            % (node["attr"]["name"], input_node[0], input_shape)
        )

        weight_shape = graph[weight_node[0]]["attr"]["attr"]["tensor_shape"]
        if len(weight_shape) != 4:
            logging.warning(
                "Failed to parse weight shape %s of node %s."
                % (str(weight_shape), node["attr"]["name"])
            )
            logging.info(node)
            return

        logging.info(
            "Get weight shape of %s from %s, input shape:%s."
            % (node["attr"]["name"], weight_node, weight_shape)
        )

        k_size = weight_shape[:2]
        cout = weight_shape[3]

        if node["attr"]["attr"]["strides"][::3] != [1, 1]:
            logging.warning(
                "Invalid strides %s of node %s."
                % (str(node["attr"]["attr"]["strides"]), node["attr"]["name"])
            )
            logging.info(node)
            return

        strides = node["attr"]["attr"]["strides"]
        dilation = node["attr"]["attr"]["dilations"]
        padding = node["attr"]["attr"]["padding"].decode("utf-8")
        logging.info(
            "Op:%s, stride:%s, dilation:%s, padding:%s."
            % (node["attr"]["name"], str(strides), str(dilation), str(padding))
        )

        kernel_extent_w = ph.get_w(dilation) * (ph.get_w(strides) - 1) + 1
        kernel_extent_h = ph.get_h(dilation) * (ph.get_h(strides) - 1) + 1

        out_shape, padding_shape = ShapeInference.get_padding_shape(
            input_shape, cout, [kernel_extent_w, kernel_extent_h], strides, padding
        )

        node["attr"]["attr"]["kernel_shape"] = copy.deepcopy(k_size)
        node["attr"]["attr"]["dilations"] = copy.deepcopy(
            node["attr"]["attr"]["dilations"][1:-1]
        )
        node["attr"]["attr"]["strides"] = copy.deepcopy(
            node["attr"]["attr"]["strides"][1:-1]
        )
        node["attr"]["attr"]["weight_shape"] = copy.deepcopy(weight_shape)
        node["attr"]["attr"]["pads"] = padding_shape

        return [input_shape], [out_shape]

    @staticmethod
    def DepthwiseConv2dNative_get_shape(graph, node):
        """
        Get shape of a DepthwiseConv2dNative operator.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        weight_node = ph.find_weights_root(graph, node)
        if len(weight_node) != 1:
            logging.warning("Failed to get shape of node %s." % (node["attr"]["name"]))
            logging.info(node)
            return

        input_node = [x for x in node["inbounds"] if x != weight_node]
        input_node = [x for x in input_node if graph[x]["attr"]["type"] != "Identity"]
        if len(input_node) != 1:
            logging.warning("Failed to get input node of %s." % (node["attr"]["name"]))
            logging.info(node)
            return

        input_shape = copy.deepcopy(graph[input_node[0]]["attr"]["output_shape"][0])
        logging.info(
            "Get input shape of %s from %s, input shape:%s."
            % (node["attr"]["name"], input_node[0], input_shape)
        )

        weight_shape = graph[weight_node[0]]["attr"]["attr"]["tensor_shape"]
        if len(weight_shape) != 4:
            logging.warning(
                "Failed to parse weight shape %s of node %s."
                % (str(weight_shape), node["attr"]["name"])
            )
            logging.info(node)
            return

        logging.info(
            "Get weight shape of %s from %s, input shape:%s."
            % (node["attr"]["name"], weight_node, weight_shape)
        )

        k_size = weight_shape[:2]
        cin = weight_shape[2]

        if node["attr"]["attr"]["strides"][::3] != [1, 1]:
            logging.warning(
                "Invalid strides %s of node %s."
                % (str(node["attr"]["attr"]["strides"]), node["attr"]["name"])
            )
            logging.info(node)
            return

        strides = node["attr"]["attr"]["strides"]
        dilation = node["attr"]["attr"]["dilations"]
        padding = node["attr"]["attr"]["padding"].decode("utf-8")

        logging.info(
            "Op:%s, stride:%s, dilation:%s, padding:%s."
            % (node["attr"]["name"], str(strides), str(dilation), str(padding))
        )

        kernel_extent_w = ph.get_w(dilation) * (ph.get_w(strides) - 1) + 1
        kernel_extent_h = ph.get_h(dilation) * (ph.get_h(strides) - 1) + 1

        out_shape, padding_shape = ShapeInference.get_padding_shape(
            input_shape, cin, [kernel_extent_w, kernel_extent_h], strides, padding
        )

        node["attr"]["attr"]["kernel_shape"] = copy.deepcopy(k_size)
        node["attr"]["attr"]["dilations"] = copy.deepcopy(
            node["attr"]["attr"]["dilations"][1:-1]
        )
        node["attr"]["attr"]["strides"] = copy.deepcopy(
            node["attr"]["attr"]["strides"][1:-1]
        )
        node["attr"]["attr"]["weight_shape"] = copy.deepcopy(weight_shape)
        node["attr"]["attr"]["pads"] = padding_shape

        return [input_shape], [out_shape]

    @staticmethod
    def Reduce_get_shape(graph, node):
        """
        Get shape of a Reduce operator.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        input_shape = graph[node["inbounds"][0]]["attr"]["output_shape"][0]
        output_shape = input_shape
        logging.info(
            "Get input shape of %s from %s, input shape:%s."
            % (node["attr"]["name"], node["inbounds"][0], output_shape)
        )

        output_shape[1] = 0
        output_shape[2] = 0

        reduction_indices = node["attr"]["attr"]["reduction_indices"]
        logging.info("Get Reduction Indices %s.", str(reduction_indices))

        reduction_cnt = 0
        for reduction in sorted(reduction_indices):
            del output_shape[reduction - reduction_cnt]
            reduction_cnt += 1

        return [input_shape], [output_shape]

    @staticmethod
    def Mean_get_shape(graph, node):
        """
        Get shape of a Mean operator.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        return ShapeInference.Reduce_get_shape(graph, node)

    @staticmethod
    def GlobalAveragePooling2D_get_shape(graph, node):
        """
        Get shape of a GlobalAveragePooling2D operator.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        return ShapeInference.Reduce_get_shape(graph, node)

    @staticmethod
    def GlobalMaxPooling2D_get_shape(graph, node):
        """
        Get shape of a GlobalMaxPooling2D operator.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        return ShapeInference.Reduce_get_shape(graph, node)

    @staticmethod
    def MatMul_get_shape(graph, node):
        """
        Get shape of a MatMul operator.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        weight_node = ph.find_weights_root(graph, node)
        if len(weight_node) != 1:
            logging.warning("Failed to get shape of node %s." % (node["attr"]["name"]))
            logging.info(node)
            return

        weight_shape = graph[weight_node[0]]["attr"]["attr"]["tensor_shape"]
        if len(weight_shape) != 2:
            logging.warning(
                "Failed to parse weight shape %s of node %s."
                % (str(weight_shape), node["attr"]["name"])
            )
            logging.info(node)
            return

        logging.info(
            "Get weight shape of %s from %s, input shape:%s."
            % (node["attr"]["name"], weight_node, weight_shape)
        )

        input_node = [x for x in node["inbounds"] if x != weight_node]
        input_node = [x for x in input_node if graph[x]["attr"]["type"] != "Identity"]
        if len(input_node) != 1:
            logging.warning("Failed to get input node of %s." % (node["attr"]["name"]))
            logging.info(node)
            return

        input_shape = copy.deepcopy(graph[input_node[0]]["attr"]["output_shape"][0])
        logging.info(
            "Get input shape of %s from %s, input shape:%s."
            % (node["attr"]["name"], input_node[0], input_shape)
        )

        if weight_shape[0] != input_shape[1]:
            logging.warning(
                "Weight shape and input shape not matched for %s."
                % (node["attr"]["name"])
            )
            logging.info(node)
            return

        output_shape = copy.deepcopy(input_shape)
        output_shape[1] = weight_shape[1]

        return [input_shape], [output_shape]

    @staticmethod
    def Reshape_get_shape(graph, node):
        """
        Get shape of a Reshape operator.
        It normally should take from the shape attribution,
        but we patch it for Pack-StrideSlice-Reshape, and
        prevent the program from dynamic inference.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        if "shape" in node["attr"]["attr"].keys():
            logging.info(
                "Shape attr find in %s op, propagate with normal.", node["attr"]["name"]
            )
            input_shape = copy.deepcopy(
                graph[node["inbounds"][0]]["attr"]["output_shape"][0]
            )
            exp_output_shape = copy.deepcopy(node["attr"]["attr"]["shape"])
        else:
            logging.info(
                "Shape attr not find in %s op, try finding the shape node.",
                node["attr"]["name"],
            )
            for in_node in node["inbounds"]:
                if graph[in_node]["attr"]["type"] == "Const":
                    exp_output_shape = copy.deepcopy(
                        graph[in_node]["attr"]["constant"]
                    )
                elif graph[in_node]["attr"]["type"] == "Pack":
                    exp_output_shape = [1] + [
                        it
                        for sl in graph[in_node]["attr"]["attr"]["constant"]
                        for it in sl
                    ]
                    logging.info(
                        "Fetched expected output shape from Pack op %s"
                        % str(exp_output_shape)
                    )
                else:
                    input_shape = copy.deepcopy(
                        graph[in_node]["attr"]["output_shape"][0]
                    )

        input_elements = abs(reduce(lambda x, y: x * y, input_shape))
        exp_output_shape_elements = abs(reduce(lambda x, y: x * y, exp_output_shape))

        if input_elements != exp_output_shape_elements:
            logging.warning(
                "Input shape %s and output shape %s not matched for %s."
                % (str(input_shape), str(exp_output_shape), node["attr"]["name"])
            )

        return [input_shape], [exp_output_shape]

    @staticmethod
    def Concat_get_shape(graph, node):
        """
        Get shape of a Concat operator.
        We add up the shape through axis.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        input_shape = []
        for in_node in node["inbounds"]:
            in_shape = graph[in_node]["attr"]["output_shape"][0]
            if in_shape != []:
                input_shape.append(in_shape)
                logging.info(
                    "Get input shape of %s from %s, input shape:%s."
                    % (node["attr"]["name"], in_node, input_shape[-1])
                )
        axis = node["attr"]["attr"]["axis"][0]

        output_shape = copy.deepcopy(input_shape[0])
        for in_shape in input_shape[1:]:
            output_shape[axis] += in_shape[axis]

        return copy.deepcopy(input_shape), [output_shape]

    @staticmethod
    def Concatenate_get_shape(graph, node):
        """
        Get shape of a Concatenate operator.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        return ShapeInference.Concat_get_shape(graph, node)

    @staticmethod
    def ConcatV2_get_shape(graph, node):
        """
        Get shape of a ConcatV2 operator.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        return ShapeInference.Concat_get_shape(graph, node)

    @staticmethod
    def Split_get_shape(graph, node):
        """
        Get shape of a Split operator.
        Also patched for Pack-StrideSlice-Split sequence.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        for in_node in node["inbounds"]:
            if graph[in_node]["attr"]["type"] == "Const":
                pass
            elif graph[in_node]["attr"]["type"] == "Pack":
                pass
            else:
                input_shape = copy.deepcopy(graph[in_node]["attr"]["output_shape"][0])

        split_dim = node["attr"]["attr"]["split_dim"][0]
        logging.info("Fetched Split dim for %s is %s.", node["attr"]["name"], split_dim)
        output_node_cnt = len(node["outbounds"])

        output_shape = copy.deepcopy(input_shape)
        output_shape[split_dim] = output_shape[split_dim] // output_node_cnt
        output_shape = copy.deepcopy([output_shape]) * output_node_cnt

        return [copy.deepcopy(input_shape)], output_shape

    @staticmethod
    def Transpose_get_shape(graph, node):
        """
        Get shape of a Transpose operator.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        for in_node in node["inbounds"]:
            if graph[in_node]["attr"]["type"] == "Const":
                perm = copy.deepcopy(graph[in_node]["attr"]["attr"]["constant"])
                logging.info("Fetched perm sequence from Const op %s" % str(perm))
            elif graph[in_node]["attr"]["type"] == "Pack":
                perm = [1] + [
                    it
                    for sl in graph[in_node]["attr"]["attr"]["constant"]
                    for it in sl
                ]
                logging.info("Fetched perm sequence from Pack op %s" % str(perm))
            else:
                input_shape = copy.deepcopy(graph[in_node]["attr"]["output_shape"][0])
                logging.info(
                    "Fetched input shape from %s,  %s" % (in_node, str(input_shape))
                )

        exp_output_shape = []
        for i in range(len(perm)):
            exp_output_shape.append(input_shape[perm[i]])

        return [input_shape], [exp_output_shape]

    @staticmethod
    def Pack_get_shape(graph, node):
        """
        Get shape of a Transpose operator.
        Patched for kernel detector.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        seq = ph.get_graph_seq(graph, [node["attr"]["name"]])[:5]
        for out_node in seq:
            if graph[out_node]["attr"]["type"] == "Reshape":
                if "input_shape" in graph[out_node]["attr"].keys():
                    return (
                        graph[out_node]["attr"]["input_shape"],
                        graph[out_node]["attr"]["input_shape"],
                    )
        return [[0, 0, 0, 0]], [[0, 0, 0, 0]]

    @staticmethod
    def StridedSlice_get_shape(graph, node):
        """
        Get shape of a Transpose operator.
        Patched for kernel detector.

        Parameters
        ----------
        graph : dict
            The Graph IR in dict format.
        node   : dict
            The node in Graph IR in dict format.
        """
        seq = ph.get_graph_seq(graph, [node["attr"]["name"]])[:5]
        for out_node in seq:
            if graph[out_node]["attr"]["type"] == "Reshape":
                if "input_shape" in graph[out_node]["attr"].keys():
                    return (
                        graph[out_node]["attr"]["input_shape"],
                        graph[out_node]["attr"]["input_shape"],
                    )
        return [[0, 0, 0, 0]], [[0, 0, 0, 0]]

    def __init__(self, model_graph, dynamic_fetcher):
        """
        Take the graph, and append output shape
        and input shape to the attributes of nodes.

        Parameters
        ----------
        model_graph : ModelGraph
            The ModelGraph IR class.
        """
        graph = model_graph.get_graph()
        seq = ph.get_graph_seq(graph, model_graph.get_graph_head())

        # Pass #1
        for node_name in seq:
            node_type = model_graph.get_node_type(node_name)
            node_get_shape_name = node_type + "_get_shape"

            # if node type find in supported ops, use faster static inference
            if node_type in self.TF_PRODCAST_MATH_OPS or \
                node_type in self.TF_PROPAGATE_MATH_OPS or \
                node_get_shape_name in dir(self) :

                if node_type in self.TF_PRODCAST_MATH_OPS:
                    input_shape, output_shape = ShapeInference.eval_prodcast(graph, graph[node_name])
                elif node_type in self.TF_PROPAGATE_MATH_OPS:
                    input_shape, output_shape = ShapeInference.propagate_shape(graph, graph[node_name])
                else:
                    input_shape, output_shape = eval("self." + node_get_shape_name)(
                        graph, graph[node_name]
                    )

            # fallback to dynamic inference
            # To be aware, dynamic inference does not process the shape at all, 
            # like removing weight shape from inputs. This may yield false prediction.
            else:
                logging.warn("%s is not supported by static inference yet." % model_graph.get_node_type(node_name))
                logging.warn("Failling back to dynamic fetcher, this may yield low inference speed.")
                input_shape, output_shape = dynamic_fetcher.get_shape_by_name(node_name)


            if output_shape is not None:
                graph[node_name]["attr"]["output_shape"] = copy.deepcopy(
                    output_shape
                )
            if input_shape is not None:
                graph[node_name]["attr"]["input_shape"] = copy.deepcopy(input_shape)
            logging.info(
                "Input shape of %s op is %s." % (node_name, str(input_shape))
            )
            logging.info(
                "Output shape of %s op is %s." % (node_name, str(output_shape))
            )


        # Pass #2
        # This is a patching for back-end, since backend extract shapes from
        # those two ops.
        for node_name in seq:
            if model_graph.get_node_type(node_name) in ["Pack", "StridedSlice"]:
                node_get_shape_name = model_graph.get_node_type(node_name) + "_get_shape"
                input_shape, output_shape = eval("self." + node_get_shape_name)(
                    graph, graph[node_name]
                )
                if output_shape is not None:
                    graph[node_name]["attr"]["output_shape"] = copy.deepcopy(
                        output_shape
                    )
                if input_shape is not None:
                    graph[node_name]["attr"]["input_shape"] = copy.deepcopy(input_shape)
                logging.info(
                    "Second Pass: Input shape of %s op is %s."
                    % (node_name, str(input_shape))
                )
                logging.info(
                    "Second Pass: Output shape of %s op is %s."
                    % (node_name, str(output_shape))
                )
