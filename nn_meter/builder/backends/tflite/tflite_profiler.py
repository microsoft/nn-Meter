# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
from ..interface import BaseProfiler


class TFLiteProfiler(BaseProfiler):
    use_gpu = None

    def __init__(self, dst_kernel_path, benchmark_model_path, graph_path='', dst_graph_path='', serial='', num_threads=1, num_runs=50, warm_ups=10):
        """
        @params:
        graph_path: graph file. path on host server
        dst_graph_path: graph file. path on android device
        kernel_path: dest kernel output file. path on android device
        benchmark_model_path: path to benchmark_model on android device
        """
        self._serial = serial
        self._graph_path = graph_path
        self._dst_graph_path = dst_graph_path
        self._dst_kernel_path = dst_kernel_path
        self._benchmark_model_path = benchmark_model_path
        self._num_threads = num_threads
        self._num_runs = num_runs
        self._warm_ups = warm_ups

    def profile(self, graph_path, preserve = False, clean = True, taskset = '70', close_xnnpack = False):
        """
        @params:
        preserve: tflite file exists in remote dir. No need to push it again.
        clean: remove tflite file after running.
        """
        model_name = os.path.basename(graph_path)
        remote_graph_path = os.path.join(self._dst_graph_path, model_name)
        taskset_cmd = f'taskset {taskset}' if taskset else ''
        kernel_cmd = f'--kernel_path={self._dst_kernel_path}' if self._dst_kernel_path else ''
        close_xnnpack_cmd = f'--use_xnnpack=false' if close_xnnpack else ''

        from ppadb.client import Client as AdbClient
        client = AdbClient(host="127.0.0.1", port=5037)
        if self._serial:
            device = client.device(self._serial)
        else:
            device = client.devices()[0]

        if not preserve:
            device.push(graph_path, remote_graph_path)
        try:
            taskset_cmd = f'taskset {taskset}' if taskset else '' 
            kernel_cmd = f'--kernel_path={self._dst_kernel_path}' if self._dst_kernel_path else ''
            close_xnnpack_cmd = f'--use_xnnpack=false' if close_xnnpack else ''
            res = device.shell(f' {taskset_cmd} {self._benchmark_model_path} {kernel_cmd} {close_xnnpack_cmd}' \
                               f' {close_xnnpack_cmd}' \
                               f' --num_threads={self._num_threads}' \
                               f' --num_runs={self._num_runs}' \
                               f' --warmup_runs={self._warm_ups}' \
                               f' --graph={remote_graph_path}' \
                               f' --enable_op_profiling=true' \
                               f' --use_gpu={"true" if self.use_gpu else "false"}')
        except:
            raise
        finally:
            if clean:
                if self._serial:
                    os.system(f"adb -s {self._serial} shell rm {remote_graph_path}")
                    os.remove(graph_path)
                else:
                    os.system(f"adb shell rm {remote_graph_path}")

        return res
