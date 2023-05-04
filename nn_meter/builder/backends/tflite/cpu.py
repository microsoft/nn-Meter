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

    def parse(self, content, model_path=None):
        self.nodes = self._parse_nodes(content, model_path)
        self.total_latency = self._parse_total_latency(content)
        if model_path:
            open(model_path.replace(".tflite", ".txt"), "w").write(content)
        return self

    def _parse_nodes(self, content, model_path):
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

        import pandas as pd
        df = pd.DataFrame(columns=('node_type', 'avg', 'name'))
        for node in nodes:
            # print({'node_type': node['node_type'], 'avg': node['avg'], 'name': node['name']})
            df.loc[len(df)] = [node['node_type'], node['avg'], node['name']]
        if model_path:
            df.to_csv(model_path.replace(".tflite", ".csv"), index=False)

        return nodes

    def _parse_total_latency(self, content):
        total_latency_regex = r'Timings \(microseconds\): count=[\d.e-]+ first=[\d.e-]+ curr=[\d.e-]+ min=[\d.e-]+ max=[\d.e-]+ avg=([\d.\+e-]+) std=([\d.\+e-]+)'

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
    
    def profile(self, converted_model, metrics = ['latency'], **kwargs):
        """ debug mode, output TFLite message to model path.
        """
        return self.parser.parse(self.profiler.profile(converted_model, **kwargs), converted_model).results.get(metrics)

