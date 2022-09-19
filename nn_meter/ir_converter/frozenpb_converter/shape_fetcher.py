# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import numpy as np
from typing import List
from nn_meter.utils.import_package import try_import_tensorflow

class ShapeFetcher:
    def __init__(self, input_graph):
        """
        Dynamically inference the node shapes.

        Parameters
        ----------
        input_graph : graph_def
            The tensorflow input graph_def file.
        """
        self.tf = try_import_tensorflow()
        self.tf.compat.v1.disable_eager_execution()

        graph = self.tf.Graph()

        with graph.as_default():
            self.tf.import_graph_def(graph_def=input_graph, name="")
        
        self.ops = graph.get_operations()
        placeholders = list(filter(lambda op: op.type == "Placeholder", self.ops))
        assert len(placeholders) == 1
        self.graph_input_tensor = placeholders[0].outputs[0]
        graph_input_tensor_shape = self.graph_input_tensor.get_shape().as_list()
        assert graph_input_tensor_shape[1] == graph_input_tensor_shape[2]
        assert graph_input_tensor_shape[3] == 3
        self.imsize = graph_input_tensor_shape[1]
        self.graph = graph

    def get_shape_by_name(self, op_name):
        """
        Get the node output shape by its name

        Parameters
        ----------
        op_name : str
            The name of the target node.
        """
        input_tensors_to_fetch = []
        output_tensors_to_fetch = []
        for op in filter(lambda op: op.name == op_name, self.ops):
            input_tensors_to_fetch.extend(op.inputs)
            output_tensors_to_fetch.extend(op.outputs)
        
        input_shape_tensors = []
        for tensor in input_tensors_to_fetch:
            input_shape_tensors.append(self.tf.compat.v1.shape(tensor))
        
        output_shape_tensors = []
        for tensor in output_tensors_to_fetch:
            output_shape_tensors.append(self.tf.compat.v1.shape(tensor))

        
        intput_shape_results = []
        output_shape_results = []
        with self.tf.compat.v1.Session(graph=self.graph) as sess:
            fake_input = np.random.randn(1, self.imsize, self.imsize, 3)
            for shape_tensor in input_shape_tensors:
                intput_shape_results.append(sess.run(
                    shape_tensor, feed_dict={self.graph_input_tensor: fake_input}
                ).tolist())
            for shape_tensor in output_shape_tensors:
                output_shape_results.append(sess.run(
                    shape_tensor, feed_dict={self.graph_input_tensor: fake_input}
                ).tolist())

        return intput_shape_results, output_shape_results