# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import subprocess
import tensorflow as tf

from ..interface import BaseBackend
from .cpu_parser import TFLiteCPUParser
from .gpu_parser import TFLiteGPUParser
from nn_meter.builder.utils import get_tensor_by_shapes


class TFLiteRunner:
    use_gpu = None

    def __init__(self, dst_kernel_path, benchmark_model_path, graph_path='', dst_graph_path='', serial='', num_threads=1, num_runs=50, warm_ups=10):
        """
        :param graph_path: graph file. path on host server
        :param dst_graph_path: graph file. path on android device
        :param kernel_path: dest kernel output file. path on android device
        :param benchmark_model_path: path to benchmark_model on android device
        """
        self._serial = serial
        self._graph_path = graph_path
        self._dst_graph_path = dst_graph_path
        self._dst_kernel_path = dst_kernel_path
        self._benchmark_model_path = benchmark_model_path
        self._num_threads = num_threads
        self._num_runs = num_runs
        self._warm_ups = warm_ups

    def load_graph(self, graph_path, dst_graph_path):
        self._graph_path = graph_path
        self._dst_graph_path = dst_graph_path

        return self

    def run(self, preserve=False, clean=False, taskset='70'):
        """
        :param preserve: tflite file exists in remote dir. No need to push it again.
        :param clean: remove tflite file after running.
        """
        base_adb_cmd = 'adb' + (f' -s {self._serial}' if self._serial != '' else '')
        push_cmd = base_adb_cmd + f' push {self._graph_path} {self._dst_graph_path}'
        taskset_cmd = ''
        if taskset:
            taskset_cmd = f'taskset {taskset}'
        run_cmd = base_adb_cmd + f' shell {taskset_cmd} {self._benchmark_model_path}' \
                                 f' --kernel_path={self._dst_kernel_path}' \
                                 f' --num_threads={self._num_threads}' \
                                 f' --num_runs={self._num_runs}' \
                                 f' --warm_umps={self._warm_ups}' \
                                 f' --graph={self._dst_graph_path}' \
                                 f' --enable_op_profiling=true' \
                                 f' --use_gpu={"true" if self.use_gpu else "false"}'
        rm_cmd = base_adb_cmd + f' shell rm {self._dst_graph_path}'

        if not preserve:
            subprocess.check_output(push_cmd, shell=True)
        try:
            res = subprocess.check_output(run_cmd, shell=True)
        except:
            raise
        finally:
            if clean:
                subprocess.check_output(rm_cmd, shell=True)
            
        res = res.decode('utf-8')

        return res


class TFLiteCPURunner(TFLiteRunner):
    use_gpu = False


class TFLiteGPURunner(TFLiteRunner):
    use_gpu = True


class TFLiteBackend(BaseBackend):
    parser_class = None
    runner_class = None

    def get_params(self):
        super().get_params()
        self.runner_kwargs.update({
            'dst_kernel_path': self.params['KERNEL_PATH'],
            'serial': self.params['DEVICE_SERIAL'],
            'benchmark_model_path': self.params['BENCHMARK_MODEL_PATH'],
        })
        self.model_dir = self.params['MODEL_DIR']
        self.remote_model_dir = self.params['REMOTE_MODEL_DIR']

    def profile(self, model, model_name, shapes=None):
        # build input shapes
        model(get_tensor_by_shapes(shapes))
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        graph_path = os.path.join(self.model_dir, model_name + '.tflite')
        open(graph_path, 'wb').write(tflite_model)
        self.runner.load_graph(graph_path, os.path.join(self.remote_model_dir, model_name + '.tflite'))
        return self.parser.parse(self.runner.run()).latency


class CPUBackend(TFLiteBackend):
    parser_class = TFLiteCPUParser
    runner_class = TFLiteCPURunner


class GPUBackend(TFLiteBackend):
    parser_class = TFLiteGPUParser
    runner_class = TFLiteGPURunner





