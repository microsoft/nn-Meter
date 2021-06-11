# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import sys
import os
import numpy as np

from nn_meter.utils.grapher_tool import Grapher
from .frozenpb_parser import FrozenPbParser
from .shape_inference import ShapeInference


class FrozenPbConverter:
    def __init__(self, file_name):
        self.graph = Grapher()

        # Parse pb to graph
        parser = FrozenPbParser(file_name)
        parser.parse_graph(self.graph)

        # Change split to more firendly scheme
        parser.fix_split_naming(self.graph)

        # Get the static shape
        ShapeInference(self.graph)

        # Strip constant and indentity nodes
        parser.strip_useless_nodes(self.graph)

    def get_flatten_grapher(self):
        def np_encoder(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    np_encoder(v)
                else:
                    if isinstance(v, np.ndarray):
                        d[k] = v.tolist()
                    if isinstance(v, (bytes, bytearray)):
                        d[k] = v.decode("utf-8")

        np_encoder(self.graph.get_graph())
        return self.graph.get_graph()
