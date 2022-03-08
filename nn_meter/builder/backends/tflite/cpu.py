# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import re
from .tflite_profiler import TFLiteProfiler
from .tflite_backend import TFLiteBackend
from ..interface import BaseParser
from nn_meter.builder.backend_meta.utils import Latency, ProfiledResults


class TFLiteCPULatencyParser(BaseParser):
    def __init__(self):
        self.nodes = []
        self.total_latency = Latency()

    def parse(self, content):
        self.nodes = self._parse_nodes(content)
        self.total_latency = self._parse_total_latency(content)
        return self

    def _parse_nodes(self, content):
        start_regex = r'[= ]*Run Order[= ]*'
        end_regex = r'[= ]*Top by Computation Time[= ]*'
        node_regex = r'\s*(\w+)\s*[\d.e-]+\s*[\d.e-]+\s*([\d.e-]+)\s*[\d.e-]+%\s*[\d.e-]+%\s*[\d.e-]+\s*1\s*(\S*)'
        flag = False

        nodes = []
        for line in content.splitlines():
            if flag:
                match = re.search(node_regex, line)
                if match:
                    nodes.append({
                        'node_type': match[1],
                        'avg': float(match[2]),
                        'name': match[3],
                    })
            if re.search(start_regex, line):
                flag = True
            if re.search(end_regex, line):
                flag = False
        
        return nodes

    def _parse_total_latency(self, content):
        total_latency_regex = r'Timings \(microseconds\): count=[\d.e-]+ first=[\d.e-]+ curr=[\d.e-]+ min=[\d.e-]+ max=[\d.e-]+ avg=([\d.e-]+) std=([\d.e-]+)'

        total_latency = Latency()
        match = re.search(total_latency_regex, content, re.MULTILINE)
        if match:
            # convert microseconds to millisecond
            total_latency = Latency(float(match[1]) / 1000, float(match[2]) / 1000)

        return total_latency
    
    @property
    def latency(self):
        return self.total_latency

    @property
    def results(self):
        results = ProfiledResults({'latency': self.latency})
        return results


class TFLiteCPUProfiler(TFLiteProfiler):
    use_gpu = False


class TFLiteCPUBackend(TFLiteBackend):
    parser_class = TFLiteCPULatencyParser
    profiler_class = TFLiteCPUProfiler
