# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import numpy as np

from .frozenpb_parser import FrozenPbParser
from .shape_inference import ShapeInference
from .shape_fetcher import ShapeFetcher
from nn_meter.utils.graph_tool import ModelGraph

class FrozenPbConverter:
    def __init__(self, file_name):
        self.model_graph = ModelGraph()

        # Parse pb to graph
        parser = FrozenPbParser(file_name)
        parser.parse_graph(self.model_graph)
        dynamic_fetcher = ShapeFetcher(parser.graph)

        # Change split to more firendly scheme
        parser.fix_split_naming(self.model_graph)

        # Get the static shape
        ShapeInference(self.model_graph, dynamic_fetcher)

        # Strip constant and indentity nodes
        parser.strip_useless_nodes(self.model_graph)

    def get_flatten_graph(self):
        def np_encoder(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    np_encoder(v)
                else:
                    if isinstance(v, np.ndarray):
                        d[k] = v.tolist()
                    if isinstance(v, (bytes, bytearray)):
                        d[k] = v.decode("utf-8")

        np_encoder(self.model_graph.get_graph())
        return self.model_graph.get_graph()
