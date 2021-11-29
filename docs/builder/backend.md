# Setup Device and Backend

To run a model and get the inference latency on the mobile device, we implement several backends. These backends will run the model and parse the command line outputs to get latency results. We provide a consistent API for such backends. Currently we provide three instance on two platforms, i.e., CPU backend, GPU backend with TFLite platform, and VPU backend with OpenVINO platform. Following we will explain how to setup the device and get connection to the backend.

nn-Meter also provides the ability to build your own customized backends, and allows users to install the customized backend as a builtin algorithm, in order for users to use the backend in the same way as nn-Meter builtin backends. To use the customized backend, users can follow the [customize backend guidance](./build_customized_backend.md). 


## Setup Device

### TFLite Android Guide

Introduction of Android Device and TFLite platform

#### 1. Install ADB and Android SDK
TODO: @Jianyu
Follow [ADB Guide](https://developer.android.com/studio/command-line/adb) to install adb on your host device.

Install android sdk/ndk

#### 2. Build TFLite Benchmark Tool
> Follow the [tensorflow official guide](https://www.tensorflow.org/lite/performance/measurement) for a more detailed guide to build and deploy `benchmark_model` onto the device.

use bazel to build a benchmark_model file. Please refer to https://dev.azure.com/v-wjiany/Personal/_git/tensorflow_modified?path=/tensorflow/lite/tools/benchmark/android/README.md&_a=preview
Follow the tensorflow official guide for a more detailed guide to build and deploy benchmark_model onto the device.
``` Bash
bazel build -c opt //tensorflow/lite/tools/benchmark:benchmark_model
```

#### 3. Setup Benckmark Tool on Device
2. Push the benchmark_model to edge device by specifying its serial.
``` Bash
adb -s <device-serial> push bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model /data/local/tmp
adb shell chmod +x /data/local/tmp/benchmark_model
```

### OpenVINO VPU Guide

Follow [OpenVINO Installation Guide](https://docs.openvinotoolkit.org/latest/installation_guides.html) to install openvino on your host.

Because some tools for VPU use a different tensorflow and python version from nn_meter, so you need to provide seperate environments for the main tool and VPU. We recommend to use [virtualenv](https://virtualenv.pypa.io/en/latest/). We use python3.6 as our test environment.

``` Bash
virtualenv openvino_env
source openvino_env/bin/activate
pip install -r docs/requirements/openvino_requirements.txt
deactivate
```

## Create Workspace and Prepare Config File

### Create Workspace
A workspace in nn-Meter is a direction to save experiment configs, test case models, and test cases json files for a group of experiments. Before connecting to the backend, a workspace folder should be created. Users could create a workspace folder by running the following command:

``` Bash
# for TFLite platform
nn-meter create --tflite-workspace <path/to/place/workspace/>

# for OpenVINO platform
nn-meter create --openvino-workspace <path/to/place/workspace/>

# for customized platform
nn-meter create --customized-workspace <backend-name> <path/to/place/workspace/>
```

After running the python code or command line statement, a workspace folder will be created and a yaml file named `backend_config.yaml` will be placed in `<workspace-path>/configs/`. Users could open `<workspace-path>/configs/backend_config.yaml` and edit the content. The config will take effect after the the config file is saved and closed.

### Prepare Configs

When connecting to backend, a series of configs should be declared and appointed by users. Specifically, for Android CPU or GPU backends, the required parameters include:

- `REMOTE_MODEL_DIR`: path to the folder (on mobile device) where temporary models will be copied to.
- `KERNEL_PATH`: path (on mobile device) where the kernel implementations will be dumped.
- `BENCHMARK_MODEL_PATH`: path (on android device) where the binary file `benchmark_model` is deployed.
- `DEVICE_SERIAL`: if there are multiple adb devices connected to your host, you need to provide the corresponding serial id. Set to `''` if there is only one device connected to your host.

For VPU backends with OpenVINO, the required parameters include:

- `OPENVINO_ENV`: path to openvino virtual environment (./docs/requirements/openvino_requirements.txt is provided)
- OPTIMIZER_PATH`: path to openvino optimizer
- `TMP_DIR`: tmp directory where temp model and profiling results will be generated
- `OPENVINO_RUNTIME_DIR`: directory to openvino runtime
- `DEVICE_SERIAL`: serial id of the device
- `DATA_TYPE`: data type of the model (e.g., fp16, fp32)

Other optional configs also include:
- `DEFAULT_INPUT_SHAPE`: default resolution and channels of your input image. Default to be #TODO
- `D1_INPUT_SHAPE`: input shapes of 1d operations like `dense`. Default to be #TODO
- `FILTERS`: filter numbers of conv and dwconv. Default to be #TODO
- `KERNEL_SIZE`: kernel size. Default to be #TODO
- `ENABLED`: rules to be tested. Default to be (I can't understand here) #TODO
- `DETAIL`: whether or not dump the detail inference time in rule files. Default to be #TODO

To edit the configs, users could open `<workspace-path>/configs/backend_config.yaml` and edit the content after creating workspace. The config will take effect after the the config file is saved and closed.

## Connect to Backend

Users could test if the connection is healthy by running
``` Bash
nn-meter connect ...
```

To apply the backend for model running, nn-Meter provides an interface `connect_backend` to initialize the backend connection. `connect_backend` has two parameters, namely `backend`, indicating name of the required backend, and `configs_path`, indicating the path to the workspace folder. 

```python
from nn_meter.builder.backend import connect_backend

workspace_path = "" # text the path to the workspace folder created in the previous step
backend = connect_backend('tflite_cpu', workspace_path=workspace_path)
...
```
