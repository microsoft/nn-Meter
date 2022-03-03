# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import subprocess
import numpy as np
import shutil
import serial

from .utils import restart
from ..interface import BaseProfiler
from nn_meter.utils.pyutils import get_pyver


class OpenVINOProfiler(BaseProfiler):

    device = None

    def __init__(self, venv, optimizer, runtime_dir, serial, graph_path='', _dst_graph_path='', data_type='FP16'):
        self._graph_path = graph_path
        self._venv = venv
        self._optimizer = optimizer
        self._dst_graph_path = _dst_graph_path
        self._runtime_dir = runtime_dir
        self._serial = serial
        self._data_type = data_type

    def load_graph(self, graph_path, dst_graph_path):
        self._graph_path = graph_path
        self._dst_graph_path = dst_graph_path

    def profile(self, shapes, retry=2):
        interpreter_path = os.path.join(self._venv, 'bin/python')
        pyver = get_pyver(interpreter_path)

        subprocess.run(
            f'{interpreter_path} {self._optimizer} '
            f'--input_model {self._graph_path} '
            f'--output_dir {self._dst_graph_path} '
            f'--data_type {self._data_type}',
            shell=True
        )

        filename = os.path.splitext(os.path.basename(self._graph_path))[0]

        input_path = os.path.join(self._dst_graph_path, 'inputs')
        if os.path.exists(input_path):
            shutil.rmtree(input_path)

        os.mkdir(input_path)
        for index, shape in enumerate(shapes):
            np.random.rand(*shape).astype('float32').tofile(os.path.join(input_path, f'input_{index}.bin'))

        output = ''

        with serial.Serial(self._serial, 115200, timeout=1) as ser:
            restart(ser)

            command = (
                f'. {os.path.join(self._venv, "bin/activate")}; '
                f'cd {self._runtime_dir}; '
                f'. {os.path.join(self._runtime_dir, "setupvars.sh")} -pyver {pyver}; '
                f'{os.path.join(self._runtime_dir, "benchmark_app")} '
                f'-i {input_path} '
                f'-m {os.path.join(self._dst_graph_path, filename + ".xml")} '
                f'-d {self.device} '
                f'-report_type detailed_counters '
                f'-report_folder {self._dst_graph_path} '
                f'-niter 50 '
                f'-nireq 1 '
                f'-api sync'
            )

            while True:
                try:
                    subprocess.run(
                        f'bash -c "{command}"',
                        shell=True,
                        timeout=30,
                    )
                    output = open(os.path.join(self._dst_graph_path, 'benchmark_detailed_counters_report.csv'), 'r').read()
                    break
                except subprocess.TimeoutExpired as e:
                    print(e)

                    if retry == 0:
                        raise e
                    print('Retrying...')
                    restart(ser)

                    retry -= 1

        return output
