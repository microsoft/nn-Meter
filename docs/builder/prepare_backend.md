# Setup Device and Backend

To get the inference latency of a model on mobile devices, we implement several backends. These backends will run the model and parse the command line outputs to get latency results. We provide a consistent API for such backends. Currently we provide three instances on two inference frameworks, i.e., CPU backend (named `"tflite_cpu"`), GPU backend (named `"tflite_gpu"`) with TFLite platform, and VPU backend (named `"openvino_vpu"`) with OpenVINO platform. Users could list all the supported backends by running
```
nn-meter --list-backends
```

Besides of the current backends, users can implement a customized backend via nn-Meter to build latency predictors for your own devices. nn-Meter  allows users to install the customized backend as a builtin algorithm, in order for users to use the backend in the same way as nn-Meter builtin backends. To use the customized backend, users can follow the [customize backend guidance](./build_customized_backend.md). 

Next, we will introduce how to setup the device and get connection to the backend.


## Setup Device

### TFLite Android Guide

TODO: Introduction of Android Device and TFLite platform

#### 1. Install ADB and Android SDK
Follow [Android Guide](https://developer.android.com/studio) to install adb on your host device.

The easiest way is to directly download Android Studio from [this page](https://developer.android.com/studio). After installing it, you will find adb at path `$HOME/Android/Sdk/platform-tools/`.


#### 2. Get TFLite Benchmark Model
The `benchmark_model` is a tool provided by [TensorFlow Lite](https://www.tensorflow.org/lite/), which can run a model and output its latency. Because nn-Meter needs to parse the text output of `benchmark_model`, a fixed version is required. For the convenience of users, we have released a modified version of `benchmark_model` based on `tensorflow==2.1`. Users could download our modified version of `benchmark_model` from [here](https://github.com/microsoft/nn-Meter/blob/dev/rule-tester/material/inference_framework_binaries/benchmark_model).

NOTE: in the situation to deal with customized test case, our `benchmark_model` is probably not suitable. Users could follow [Official Guidance](https://www.tensorflow.org/lite/performance/measurement) to build benchmark tool with new version `TensorFlow Lite`. Meanwhile, the class of `LatencyParser` may need to be refined. We are working to release the source code of this modified version.

#### 3. Setup Benckmark Tool on Device

Push the `benchmark_model` to edge device by specifying its serial (if any).

``` Bash
adb [-s <device-serial>] push bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model /data/local/tmp

# add executable permission to benchmark_model
adb shell chmod +x /data/local/tmp/benchmark_model
```

### OpenVINO VPU Guide

Follow [OpenVINO Installation Guide](https://docs.openvinotoolkit.org/latest/installation_guides.html) to install openvino on your host.

Since some tools for VPU use a different tensorflow and python version from nn_meter, you need to provide seperate environments for the main tool and VPU. We recommend to use [virtualenv](https://virtualenv.pypa.io/en/latest/). We use python3.6 as our test environment.

``` Bash
virtualenv openvino_env
source openvino_env/bin/activate
pip install -r docs/requirements/openvino_requirements.txt
deactivate
```

## <span id="prepare-configuration-file"> Prepare Configuration File </span>

When connecting to backend, a series of configs should be declared and appointed by users. After creating workspace folder ([Workspace Guidance](overview.md#create-workspace)), a yaml file named `backend_config.yaml` will be placed in `<workspace-path>/configs/`. Users could open `<workspace-path>/configs/backend_config.yaml` and edit the content to change configuration.

Specifically, for Android CPU or GPU backends, the required parameters include:

- `REMOTE_MODEL_DIR`: path to the folder (on mobile device) where temporary models will be copied to.
- `KERNEL_PATH`: path (on mobile device) where the kernel implementations will be dumped.
- `BENCHMARK_MODEL_PATH`: path (on android device) where the binary file `benchmark_model` is deployed.
- `DEVICE_SERIAL`: if there are multiple adb devices connected to your host, you need to provide the corresponding serial id. Set to `''` if there is only one device connected to your host.

For VPU backends with OpenVINO, the required parameters include:

- `OPENVINO_ENV`: path to openvino virtual environment (./docs/requirements/openvino_requirements.txt is provided)
- `OPTIMIZER_PATH`: path to openvino optimizer
- `OPENVINO_RUNTIME_DIR`: directory to openvino runtime
- `DEVICE_SERIAL`: serial id of the device
- `DATA_TYPE`: data type of the model (e.g., fp16, fp32)

Users could open `<workspace-path>/configs/backend_config.yaml` and edit the content. After completing configuration, users could initialize workspace in `builder_config` module before connecting backend:

```python
from nn_meter.builder import builder_config

# initialize builder config with workspace
builder_config.init(
    workspace_path="path/to/workspace/folder"
) # change the text to required platform type and workspace path
```

Note: after running ``builder_config.init``, the config are loaded already. If users want to update config, after the updated config file is saved and closed, the config will take effect after reload config space by running ``builder_config.init`` again.

## Connect to Backend

Users could test if the connection is healthy by running

``` Bash
nn-meter connect --backend <backend-name> --workspace <path/to/workspace>
```

If the connection is successful, there will be a message saying:

``` text
(nn-Meter) hello backend !
```

To apply the backend for model running, nn-Meter provides an interface `connect_backend` to initialize the backend connection. When using `connect_backend`, name of the required backend needs to be declared. 

```python
# initialize workspace in code
workspace_path = "/path/to/workspace/" 
from nn_meter.builder import builder_config
builder_config.init(workspace_path)

# connect to backend
from nn_meter.builder.backends import connect_backend
backend = connect_backend(backend_name='tflite_cpu')
...
```
Users could follow [this example](../../examples/nn-meter_builder_with_tflite.ipynb) to further know about our API.


# <span id="build-customized-backend"> Build Customized Backend </span>

## Prepare Customized Backend Class

nn-Meter provide API for users to customize their own backend. Here we describe the implementation of `BaseBackend` first. We define the base of all backend in `nn_meter.builder.backend.BaseBackend`. There are following methods in this base class:

- `runner_class`: should be a subclass inherit form `nn_meter.builder.backend.BaseRunner` to specify the running command of the backend. A runner contains commands to push the model to mobile device, run the model on the mobile device, get stdout from the mobile device, and related operations. In the implementation of a runner, an interface of ``Runner.run()`` is required. Users need to modify this **at the most time**.

    - `run`: Main steps of ``Runner.run()`` includes 1) push the model file to edge devices, 2) run models in required times and get back running results. Return the running results on edge device.

- `parser_class`: should be a subclass inherit form `nn_meter.builder.backend.BaseParser` to parse the profiled results. A parser parses the stdout from runner and get required metrics. In the implementation of a parser, interface of `Parser.parse()` and property of `Parser.results()` are required. Considering there will always be different for new backend, users need to modify this **all the time**.
    
    - `parse`: a string parser to parse profiled results value from the standard output of devices runner. This method should return the instance class itself.

    - `results`: warp the parsed results by ``ProfiledResults`` class from ``nn_meter.builder.backend_meta.utils`` and return the parsed results value.

- `update_configs`: update the config parameters for the backend. Users need to modify this **all the time**.

- `convert_model`: convert the Keras model instance to the type required by the backend inference. For **some time you will** need to modify this.

- `profile`: load model by model file path and run ``self.profile()``. nn-Meter only support latency for metric by now. Users may provide other
        metrics in their customized backend. At the **most time you won't** need to modify this.

- `profile_model_file`: load model by model file path and run ``self.profile()``. At the **most time you won't** need to modify this.

- `test_connection`: check the status of backend interface connection. For **some time you won't** need to implement this as it is for testing only.

Here is an example to create a new backend class:

```python
from nn_meter.builder.backend import BaseBackend, BaseParser, BaseRunner

class MyParser(BaseParser): ...
class MyRunner(BaseRunner): ...

class MyBackend(BaseBackend):
    parser_class = MyParser
    runner_class = MyRunner
```

Besides, nn-Meter also provide TFLite backend (`nn_meter.builder.backend.TFLiteBackend`), and OpenVINO backend (`nn_meter.builder.backend.OpenVINOBackend`), in case if users want to create new device instance based on TFLite or OpenVINO. By inheriting these two class, users could reuse some methods, such as `convert_model`, `profile`, and `test_connection`.

Here is an example to inherit `TFLiteBackend` and create backend named `my_tflite`:

```python
from nn_meter.builder.backend import TFLiteBackend, TFLiteRunner, BaseParser

class MyParser(BaseParser): ...
class MyRunner(TFLiteRunner): ...

class MyTFLiteBackend(TFLiteBackend):
    parser_class = MyParser
    runner_class = MyRunner
```

## Register Backend to nn-Meter

### Step 1: Create a Package for the Customized Backend

After preparing the backend class, users should also prepare a default config file in yaml format if there is any modifiable configs. Therefore, after the registration of customized backend, the config file will be copied to workspace when running `nn-meter create --customized-workspace`. Users could refer to [the Configuration of TFLite and OpenVINO](#prepare-configuration-file) as a reference. nn-Meter suggests users to gather all code of backend and default config file in a package with a fixed location. The folder should contain all dependent classes, such as `Parser` and `Runner`. A folder will be treated as a package with a `__init__.py` file added. Here is a demo of folder structure:

``` text
./customized_backend/
├── __init__.py
├── backend.py
├── utils.py
└── default_config.yaml
```

The interface of customized backend class are stored in `./customized_backend/backend.py`. In this demo, the content in `backend.py` includes:

``` python
import logging
from nn_meter.builder.backends import BaseBackend, BaseParser, BaseRunner

class MyParser(BaseParser): ...

class MyRunner(BaseRunner): ...

class MyBackend(BaseBackend):
    parser_class = MyParser
    runner_class = MyRunner

    def __init__(self, config):
        pass

    def test_connection(self):
        """check the status of backend interface connection
        """
        ...
        logging.keyinfo("hello backend !")
```

### Step 2: Prepare Meta File

Create a yaml file with following keys as meta file:

- `builtin_name`: builtin name used in nn-Meter configuration file to call the customized backend, such as `"my_backend"`.

- `package_location`: the absolute path of the package.

- `class_module`: the module of the backend class, in this example is `backend`, representing `backend.py`.

- `class_name`: the backend class name, in this example is `MyBackend`.

- `defaultConfigFile`: the absolute path of the default configuration file. 

Following is an example of the yaml file:

```yaml
builtin_name: my_backend
package_location: /home/USERNAME/working/customized_backend
class_module: backend
class_name: MyBackend
defaultConfigFile: /home/USERNAME/working/customized_backend/default_config.yaml
```

### Step 3: Register Customized Backend into nn-Meter

Run the following command to register customized backend into nn-Meter:

``` bash
nn-meter register --backend path/to/meta/file
```
If the registration success, nn-Meter will show:
``` text
(nn-Meter) Successfully register backend my_backend.
```

When registering, nn-Meter will test whether the module can be imported first. If the registration success is not successful, please check the package according to the error information.

After backend registration, users can view all backends by running:
``` bash
nn-meter --list-backends
```
```text
(nn-Meter) Supported backends: ('*' indicates customized backends)
(nn-Meter) [Backend] tflite_cpu
(nn-Meter) [Backend] tflite_gpu
(nn-Meter) [Backend] openvino_vpu
(nn-Meter) [Backend] * my_backend
```

Note: the package of customized backend must be retained in a fixed path as registered one. Otherwise may cause error when calling the registered module.

### Step 4: Test the Registered Backend

After registration, users could create customized workspace according to the customized backend:

``` bash
nn-meter create --customized-workspace <workspace-path> --backend my_backend 
```
``` text
(nn-Meter) Workspace <workspace-path> for customized platform has been created. Users could edit experiment config in <workspace-path>/configs/.
```

Users could edit experiment configuration file in `<workspace-path>/configs/backend_config.yaml`, and test the connection to the registered backend by running:

``` bash
nn-meter connect --backend my_backend --workspace <workspace-path>
```
```
(nn-Meter) hello backend !
```

## Use the Customized Backend in Experiment

After registration, users could get access to the customized backend by calling its builtin name:

``` python
# initialize builder config with workspace
from nn_meter.builder import builder_config
builder_config.init(workspace_path="...") # the path of workspace

# connect to backend
from nn_meter.builder.backends import connect_backend
backend = connect_backend(backend_name='my_backend')
```

## Manage the Registered Backend

Users can view all builtin and registered backends by running:

``` bash
nn-meter --list-backends
```
```text
(nn-Meter) Supported backends: ('*' indicates customized backends)
(nn-Meter) [Backend] tflite_cpu
(nn-Meter) [Backend] tflite_gpu
(nn-Meter) [Backend] openvino_vpu
(nn-Meter) [Backend] * my_backend
```

Besides, users could unregister the backend by calling its name in command:

``` bash
nn-meter unregister --backend my_backend
```
``` text
(nn-Meter) Successfully unregister my_backend.
```

After unregister the backend, "my_backend" will be removed from the backend list:

``` bash
nn-meter --list-backends
```
``` text
(nn-Meter) Supported backends: ('*' indicates customized backends)
(nn-Meter) [Backend] tflite_cpu
(nn-Meter) [Backend] tflite_gpu
(nn-Meter) [Backend] openvino_vpu
```