# Setup Device and Backend

To run a model on the device, we implement several backends. These backends will run the model and parse the command line outputs to get latency results. We provide a consistent API for such backends. You can inherit the `BaseBackend` and implement a backend for your own device that's not included here.

Currently we support TFLite on CPU, GPU and OpenVINO on VPU. Following we will explain how to setup the device and backend.

## Android Guide

### Prepare Android Device

> Follow [ADB Guide](https://developer.android.com/studio/command-line/adb) to install adb on your host device.

> Follow the [tensorflow official guide](https://www.tensorflow.org/lite/performance/measurement) for a more detailed guide to build and deploy `benchmark_model` onto the device.


Download tensorflow source code and type following lines in tensorflow root folder:
```
bazel build -c opt //tensorflow/lite/tools/benchmark:benchmark_model
adb push bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model /data/local/tmp
adb shell chmod +x /data/local/tmp/benchmark_model
```

### Usage

Use api `get_backend` to initialize the Android CPU or GPU backend.

```python
backend = get_backend('tflite_cpu', {
    'MODEL_DIR': '~/models',  # directory on host to save temporary tflite models
    'REMOTE_MODEL_DIR': '/data/local/tmp/models',  # directory on mobile phone to place models
    'KERNEL_PATH': '/data/local/tmp/kernels.cl',  # directory on mobile phone where kernel code files will be generated
    'BENCHMARK_MODEL_PATH': '/data/local/tmp/benchmark_model',  # path to bin of `benchmark_model`
    'DEVICE_SERIAL': '',  # serial id of the device. set to '' if there is only one device connected to your host
})
```

### Modified Build for TFLite GPU

TODO

## VPU Guide

### Prerequisits

#### OpenVINO

Follow [OpenVINO Installation Guide](https://docs.openvinotoolkit.org/latest/installation_guides.html) to install openvino on your host.

#### Python Environments

Because some tools for VPU use a different tensorflow and python version from nn_meter, so you need to provide seperate environments for the main tool and VPU. We recommend to use [virtualenv](https://virtualenv.pypa.io/en/latest/). We use python3.6 as our test enviroment.

```
virtualenv movidius_env
source movidius_env/bin/activate
pip install -r movidius_requirements.txt
deactivate
```

### Usage

Suppose you place the python environments at `~/movidius_env`.

```python
backend = get_backend('vpu', {
    'MOVIDIUS_ENV': '~/movidius_env',
    'OPTIMIZER_PATH': '/data/openvino_2019.2.242/deployment_tools/model_optimizer/mo_tf.py',
    'TMP_DIR': '~/models',
    'OPENVINO_RUNTIME_DIR': '/data/openvino_2019.2.242/bin',
    'DEVICE_SERIAL': '/dev/ttyUSB4',
    'DATA_TYPE': 'FP16',
})
```
