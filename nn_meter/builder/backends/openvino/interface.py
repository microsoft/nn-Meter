# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import tensorflow as tf
import subprocess
import numpy as np
import shutil

from .utils import serial
from .vpu_parser import VPUParser
from ..interface import BaseBackend
from .utils import keras_model_to_frozenpb
from .frozenpb_patcher import patch_frozenpb
from .utils import restart
from nn_meter.utils.path import get_filename_without_ext
from nn_meter.utils.pyutils import get_pyver


class VPURunner:
    def __init__(self, venv, optimizer, runtime_dir, serial, graph_path='', output_dir='', data_type='FP16'):
        self._graph_path = graph_path
        self._venv = venv
        self._optimizer = optimizer
        self._output_dir = output_dir
        self._runtime_dir = runtime_dir
        self._serial = serial
        self._data_type = data_type

    def load_graph(self, graph_path, output_dir):
        self._graph_path = graph_path
        self._output_dir = output_dir

    def run(self, shapes, retry=2):
        interpreter_path = os.path.join(self._venv, 'bin/python')
        pyver = get_pyver(interpreter_path)

        subprocess.run(
            f'{interpreter_path} {self._optimizer} '
            f'--input_model {self._graph_path} '
            f'--output_dir {self._output_dir} '
            f'--data_type {self._data_type}',
            shell=True
        )

        filename = os.path.splitext(os.path.basename(self._graph_path))[0]

        input_path = os.path.join(self._output_dir, 'inputs')
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
                f'-m {os.path.join(self._output_dir, filename + ".xml")} '
                f'-d MYRIAD '
                f'-report_type detailed_counters '
                f'-report_folder {self._output_dir} '
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
                    output = open(os.path.join(self._output_dir, 'benchmark_detailed_counters_report.csv'), 'r').read()
                    break
                except subprocess.TimeoutExpired as e:
                    print(e)

                    if retry == 0:
                        raise e
                    print('Retrying...')
                    restart(ser)

                    retry -= 1

        return output


class VPUBackend(BaseBackend):
    parser_class = VPUParser
    runner_class = VPURunner

    def get_params(self):
        super().get_params()
        self.runner_kwargs.update({
            'venv': self.params['MOVIDIUS_ENV'],
            'optimizer': self.params['OPTIMIZER_PATH'],
            'runtime_dir': self.params['OPENVINO_RUNTIME_DIR'],
            'serial': self.params['DEVICE_SERIAL'],
            'data_type': self.params['DATA_TYPE'],
        })
        self.tmp_dir = self.params['TMP_DIR']
        self.venv = self.params['MOVIDIUS_ENV']

    def profile(self, model, model_name, shapes):
        model_tmp_dir = os.path.join(self.tmp_dir, model_name)
        pb_path, _ = keras_model_to_frozenpb(model, model_tmp_dir, model_name, shapes)
        patched_pb_path = patch_frozenpb(pb_path, os.path.join(self.venv, 'bin/python'))
        self.runner.load_graph(patched_pb_path, model_tmp_dir)
        return self.parser.parse(self.runner.run(shapes)).latency

    def profile_model_file(self, model_path, shapes):
        model_name = get_filename_without_ext(model_path)
        model = tf.keras.models.load_model(model_path)
        return self.profile(model, model_name, shapes)

