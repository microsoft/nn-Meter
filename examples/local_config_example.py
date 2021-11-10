import os

ENABLED = ['MON']

HOME_DIR = os.path.expanduser('~')

BASE_DIR = os.path.dirname(__file__)

BACKENDS = {
    'tflite_gpu': {
        'ENGINE': 'backends.tflite_gpu',
        'PARAMS': {
            'MODEL_DIR': '/data1/datasets_pad', #os.path.join(HOME_DIR, "benchmarks/models/tflite"),
            'REMOTE_MODEL_DIR': '/mnt/sdcard/tflite_bench',
            'KERNEL_PATH': '/mnt/sdcard/tflite_bench/kernel.cl',
            'BENCHMARK_MODEL_PATH': '/data/local/tmp/benchmark_model_fixed_group_size',
            'DEVICE_SERIAL': '5e6fecf',
        },
        'ENABLED': True,
    },
    'tflite_cpu': {
        'ENGINE': 'backends.tflite_cpu',
        'PARAMS': {
            'MODEL_DIR': os.path.join(HOME_DIR, "benchmarks/models/tflite"),
            'REMOTE_MODEL_DIR': '/mnt/sdcard/tflite_bench',
            'KERNEL_PATH': '/mnt/sdcard/tflite_bench/kernel.cl',
            'BENCHMARK_MODEL_PATH': '/data/local/tmp/benchmark_model_fixed_group_size',
            'DEVICE_SERIAL': '5e6fecf',
        },
        'ENABLED': True,
    },
    'vpu': {
        'ENGINE': 'backends.vpu',
        'PARAMS': {
            'OPENVINO_ENV': os.path.join(BASE_DIR, 'openvino_env'),
            'OPTIMIZER_PATH': '/data/openvino_2019.2.242/deployment_tools/model_optimizer/mo_tf.py',
            'TMP_DIR': os.path.join(HOME_DIR, 'benchmarks/openvino'),
            'OPENVINO_RUNTIME_DIR': '/data/openvino_2019.2.242/bin',
            'DEVICE_SERIAL': '/dev/ttyUSB4',
            'DATA_TYPE': 'FP16',
        },
        'ENABLED': False,
    }
}

OUTPUT_PATH = './gpuconv-bn-relu-test_tflite_gpu.csv'

DETAIL = True
