# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import re
from .openvino_profiler import OpenVINOProfiler
from .openvino_backend import OpenVINOBackend
from ..interface import BaseParser
from nn_meter.builder.backend_meta.utils import Latency, ProfiledResults


class OpenVINOVPULatencyParser(BaseParser):

    def parse(self, content):
        self.layers = self._parse_layers(content)
        self.comp_layer_latency = sum(
            Latency(layer['realtime'])
            for layer in self.layers
            if layer['layer_name'] != '<Extra>'
        )
        return self

    def _parse_layers(self, content):
        layer_regex = r'^([^;]+);([^;]+);([^;]+);([^;]+);([^;]+);([^;]+);$'
        layers = []
        for match in re.findall(layer_regex, content, re.MULTILINE):
            try:
                layers.append({
                    'layer_name': match[0],
                    'exec_status': match[1],
                    'layer_type': match[2],
                    'exec_type': match[3],
                    'realtime': float(match[4]),
                    'cputime': float(match[5]),
                })
            except:
                pass
        return layers

    @property
    def latency(self):
        return self.comp_layer_latency
    
    @property
    def results(self):
        results = ProfiledResults({'latency': self.latency})
        return results


class OpenVINOVPUProfiler(OpenVINOProfiler):
    device = "MYRIAD"


class OpenVINOVPUBackend(OpenVINOBackend):
    parser_class = OpenVINOVPULatencyParser
    profiler_class = OpenVINOVPUProfiler
