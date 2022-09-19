# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import re
from .tflite_profiler import TFLiteProfiler
from .tflite_backend import TFLiteBackend
from ..interface import BaseParser
from nn_meter.builder.backend_meta.utils import Latency, ProfiledResults


class TFLiteGPULatencyParser(BaseParser):
    def __init__(self):
        self.kernels = []
        self.realtime = 0
        self.kernel_sum = 0
        self.block_name = ''
        self.raw_content = ''
        self.before_fused_graph = ''
        self.after_fused_graph = ''

    def parse(self, content):
        result = self._parse_time(content)
        kernel_operation_map = self._parse_kernel_name(content)
        work_size = self._parse_work_size(content)
        self.realtime, self.block_name = self._parse_block(content)
        self.kernel_sum = sum(value[0] for key, value in result.items())
        self.kernels = [{}] * len(result)
        self.before_fused_graph, self.after_fused_graph = self._parse_graph(content)
        self.comp_avg, self.comp_std = self._parse_comp_time(content)
        self.nodes = self._parse_node_cpu_time(content)
        self.errors = self._parse_error(content)
        for key, value in result.items():
            self.kernels[key] = {
                'avg': value[0],
                'std': value[1],
                'work_size': work_size[key],
                'name': kernel_operation_map[key],
            }

        self.comp_kernel_latency = sum((Latency(kernel['avg'], kernel['std']) for kernel in self.kernels if kernel['name'] != 'to/from tensor'), Latency())

        self.raw_content = content

        return self

    @staticmethod
    def resolve_name(name):
        name = name.split(' ')
        if 'linked' in name:
            ops = []
            name = [x for x in name if x != ':' and x != 'linked']
            for i in range(0, len(name), 2):
                ops.append(name[i])
            return ops
        else:
            return [name[0]]

    @property
    def latency(self):
        """
        On GPU, we currently decide to use kernel_sum instead of realtime (block) as latency
        """
        return self.comp_kernel_latency

    def _parse_kernel_name(self, content):
        kernel_name_regex = r'kernel_name\[(\d+)\]=(.*)'
        kernel_operation_map = {}

        for line in content.splitlines():
            match = re.search(kernel_name_regex, line)
            if match:
                index = int(match[1])
                kernel_operation_map[index] = match[2]

        return kernel_operation_map

    def _parse_block(self, content):
        node_regex = r'\s+\w+\s+[\d.e-]+\s+[\d.e-]+\s+([\d.e-]+)[\s\d.%]+(\S+)'

        realtime = 0
        block_name = ''
        for line in content.splitlines():
            match = re.search(node_regex, line)
            if match:
                realtime = float(match[1])
                block_name = match[2]
                break

        return realtime, block_name

    def _parse_time(self, content):
        kernel_regex = r'\w+\[(\d+)\]\w+=([\d.e-]+) \w+\[(\d+)\]\w+=([\d.e-]+) ' \
                       r'\w+\[(\d+)\]\w+=([\d.e-]+) \w+\[(\d+)\]\w+=([\d.e-]+)'
        result = {}

        for line in content.splitlines():
            match = re.search(kernel_regex, line)
            if match:
                index = int(match[1])
                avg_ms = float(match[2])
                std_ms = float(match[4])
                result[index] = (avg_ms, std_ms)

        return result

    def _parse_work_size(self, content):
        work_size_regex = r'local_work_size\[(\d+)\]=([\d,]+)'
        work_size = {}

        for line in content.splitlines():
            match = re.search(work_size_regex, line)
            if match:
                index = int(match[1])
                work_size[index] = match[2]

        return work_size

    def _parse_graph(self, content):
        before_fused_regex = r'\[Before Fused\](.*)\[end\]'
        before_fused_pattern = re.compile(before_fused_regex, re.DOTALL)

        before_fused_graph = ''
        match = before_fused_pattern.search(content)
        if match:
            before_fused_graph = match[1]

        after_fused_regex = r'\[After Fused\](.*)\[end\]'
        after_fused_pattern = re.compile(after_fused_regex, re.DOTALL)

        after_fused_graph = ''
        match = after_fused_pattern.search(content)
        if match:
            after_fused_graph = match[1]

        return before_fused_graph, after_fused_graph

    def _parse_comp_time(self, content):
        comp_time_regex = r'comp_avg_ms=([\d.e-]+) comp_std_ms=([\d.e-]+)'
        comp_avg, comp_std = 0, 0

        for line in content.splitlines():
            match = re.search(comp_time_regex, line)
            if match:
                comp_avg = float(match[1])
                comp_std = float(match[2])

        return comp_avg, comp_std

    def _parse_node_cpu_time(self, content):
        node_regex = r'(\w+)\s+[\d]+\s+([\d.e-]+)\s+[\d.%]+\s+[\d.%]+\s+[\d.e-]+\s+\d+'
        nodes = {}

        for line in content.splitlines():
            match = re.search(node_regex, line)
            if match:
                nodes[match[1]] = float(match[2])

        return nodes

    def _parse_error(self, content):
        error_regex = r'ERROR: (.*)'
        errors = []

        for line in content.splitlines():
            match = re.search(error_regex, line)
            if match:
                errors.append(match[1])

        return errors

    @property
    def results(self):
        results = ProfiledResults({'latency': self.latency})
        return results


class TFLiteGPUProfiler(TFLiteProfiler):
    use_gpu = True


class TFLiteGPUBackend(TFLiteBackend):
    parser_class = TFLiteGPULatencyParser
    profiler_class = TFLiteGPUProfiler
