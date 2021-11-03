# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import re
from nn_meter.builder.utils.latency import Latency


class TFLiteCPUParser:
    def __init__(self):
        self.nodes = []
        self.total_latency = Latency()

    def parse(self, output):
        self.nodes = self._parse_nodes(output)
        self.total_latency = self._parse_total_latency(output)
        return self

    def _parse_nodes(self, output):
        start_regex = r'[= ]*Run Order[= ]*'
        end_regex = r'[= ]*Top by Computation Time[= ]*'
        node_regex = r'\s*(\w+)\s*[\d.e-]+\s*[\d.e-]+\s*([\d.e-]+)\s*[\d.e-]+%\s*[\d.e-]+%\s*[\d.e-]+\s*1\s*(\S*)'
        flag = False

        nodes = []
        for line in output.splitlines():
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

    def _parse_total_latency(self, output):
        total_latency_regex = r'Timings \(microseconds\): count=[\d.e-]+ first=[\d.e-]+ curr=[\d.e-]+ min=[\d.e-]+ max=[\d.e-]+ avg=([\d.e-]+) std=([\d.e-]+)'

        total_latency = Latency()
        match = re.search(total_latency_regex, output, re.MULTILINE)
        if match:
            total_latency = Latency(float(match[1]), float(match[2]))

        return total_latency
    
    @property
    def latency(self):
        return self.total_latency
        