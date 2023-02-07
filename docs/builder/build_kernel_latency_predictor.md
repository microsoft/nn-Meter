# Build Kernel Latency Predictor

## Step1: Prepare Backends and Create Workspace

The first step to build kernel latency predictor is to prepare backends and create workspace. Users could follow guidance [Prepare Backends](./prepare_backend.md) and [Create Workspace](./overview.md#create-workspace) for this step.

After creating the workspace, a yaml file named `predictorbuild_config.yaml` will be placed in `<workspace-path>/configs/`. The predictor build configs includes:

- `DETAIL`: Whether to attach detail information to the json output, such as the shape and configuration information in profiled results. Default value is `FALSE`.
- `IMPLEMENT`: The code implementation, could be chosen from [`tensorflow`, `torch`].
- `BATCH_SIZE`: The batch size in kernel profiling. Default value is 1.
- `KERNELS`: The training parameters for each kernel. By default, nn-Meter set 16 kernels, including "conv-bn-relu", "dwconv-bn-relu", "maxpool", "avgpool", "fc", "concat", "split", "channelshuffle", "se", "global-avgpool", "bnrelu", "bn", "hswish", "relu", "addrelu", "add". For each type of kernel, the parameters includes:
  - `INIT_SAMPLE_NUM`: the data size for predictor initialization.
  - `FINEGRAINED_SAMPLE_NUM`: the data size for adaptive sampling. For each data with error higher than error_threshold, number of `FINEGRAINED_SAMPLE_NUM` data will be generated based the the large error data. Defaults to 20.
  - `ITERATION`: the iteration for sampling and training. Predictor training based on initial sampling is regarded as iteration 1, thus `iteration == 2` means one iteration for adaptive sampling.
  - `ERROR_THRESHOLD`: the threshold of large error. Defaults to 0.1.

Users could open `<workspace-path>/configs/predictorbuild_config.yaml` and edit the content. After completing configuration, users could initialize workspace in `builder_config` module before building the kernel latency predictor:

```python
from nn_meter.builder import builder_config

# initialize builder config with workspace
builder_config.init(
    workspace_path="path/to/workspace/folder"
) # change the text to required platform type and workspace path
```

Note: after running ``builder_config.init``, the config are loaded already. If users want to update config, after the updated config file is saved and closed, the config will take effect after reload config space by running ``builder_config.init`` again.

## Step2: Config Sampling From Prior Distribution

To learn the relationship between configurations and latency, we need to generate a training set (i.e., variously configured kernels and the latencies) for regression. While it's unfeasible to sample and measure all the configurations for all kernels, a direct method is random sampling.

The first step is sampling configuration values from the prior distribution, which is inferred from the existing models. Based on our kernel model, there are generally 6 configuration values, including height and width (`"HW"`), input channel (`"CIN"`), output channel (`"COUT"`), kernel size (`"KERNEL_SIZE"`), strides (`"STRIDES"`), and kernel size for pooling layer (`"POOL_STRIDES"`). We sampling the configuration based on the prior distribution and adapt the value to common valid values. That is, height and weight are verified to value from `[1, 3, 7, 14, 28, 56, 112, 224]`, kernel size to `[1, 3, 5, 7]`, strides to `[1, 2, 4]`, and kernel size for pooling layer to `[2, 3]`. We stored the prior knowledge of existing models as csv files in `nn_meter/builder/kernel_predictor_builder/data_sampler/prior_config_lib/`.

## Step 3: Generate and Profile Kernel Model by Configs

The second step is generating and profiling kernel model by configurations. nn-Meter supports both implementation of Tensorflow and PyTorch kernels. Users could switch the kernel implementation between Tensorflow and PyTorch by editing configuration `IMPLEMENT` in `<workspace-path>/configs/predictorbuild_config.yaml`. Here we use Tensorflow implementation and `"tflite_cpu"` backend as an example.

Currently, the kernel blocks and corresponding configurations supported by nn-Meter include:

(conv related kernels)

- `"conv-bn-relu"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `STRIDES`
- `"conv-bn-relu6"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `STRIDES`
- `"conv-bn"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `STRIDES`
- `"conv-relu"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `STRIDES`
- `"conv-relu6"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `STRIDES`
- `"conv-hswish"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `STRIDES`
- `"conv-block"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `STRIDES`
- `"conv-bn-hswish"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `STRIDES`

(dwconv related kernels, where config "CIN" will always be the same as "COUT")
- `"dwconv-bn"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `STRIDES`
- `"dwconv-relu"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `STRIDES`
- `"dwconv-relu6"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `STRIDES`
- `"dwconv-bn-relu"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `STRIDES`
- `"dwconv-bn-relu6"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `STRIDES`
- `"dwconv-block"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `STRIDES`
- `"dwconv-bn-hswish"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `STRIDES`

(other kernels)
- `"maxpool"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `POOL_STRIDES`
- `"avgpool"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `POOL_STRIDES`
- `"fc"`: `CIN`, `COUT`
- `"concat"`: `HW`, `CIN1`, `CIN2`, `CIN3`, `CIN4`
- `"split"`: `HW`, `CIN`
- `"channelshuffle"`: `HW`, `CIN`
- `"se"`: `HW`, `CIN`
- `"global-avgpool"`: `HW`, `CIN`
- `"bnrelu"`: `HW`, `CIN`
- `"bn"`: `HW`, `CIN`
- `"hswish"`: `HW`, `CIN`
- `"relu"`: `HW`, `CIN`
- `"addrelu"`: `HW`, `CIN`, `CIN`
- `"add"`: `HW`, `CIN`, `CIN`

The first and second step are implemented by `nn_meter.builder.nn_meter_builder.sample_and_profile_kernel_data`. Here is an example:

``` python
from nn_meter.builder import builder_config 
from nn_meter.builder.nn_meter_builder import sample_and_profile_kernel_data
workspace = "/path/to/workspace/"
builder_config.init(workspace)

kernel_type = "conv-bn-relu"
sample_num = 10
backend = "tflite_cpu"
mark = "test"

kernel_data = sample_and_profile_kernel_data(kernel_type, sample_num=sample_num,
                                             backend=backend, sampling_mode='prior', mark=mark)
```

The generated models are saved in `<workspace-path>/predictor_build/kernels`, and the configuration information and profiled results are dumped in json file to `<workspace-path>/predictor_build/results/<kernel_type>.json` and `<workspace-path>/predictor_build/results/profiled_<kernel_type>.json`.

Note: sometimes the number of sampling kernel data is smaller than the value of `sample_num`. It is natural since nn-Meter will remove duplicates sample when generating kernel data.

The method `sample_and_profile_kernel_data` is composed by three main steps, `generate_config_sample`, `convert_models`, and `profile_models`. Here is an example as a decomposition of `sample_and_profile_kernel_data`. Users could choose the decomposed interfaces if needed.

``` python
from nn_meter.builder.backends import connect_backend
from nn_meter.builder.kernel_predictor_builder import generate_config_sample
from nn_meter.builder import builder_config, convert_models, profile_models  
workspace = "/path/to/workspace/"
builder_config.init(workspace)

backend = connect_backend(backend_name="tflite_cpu")

kernel_type = "conv-bn-relu"
sample_num = 10
mark = "test"

# sample configs for kernel and generate models
models = generate_config_sample(kernel_type, sample_num, mark=mark,
                                sampling_mode="prior")

# convert the model to the needed format by backend, in order to increase efficiency when profiling on device.
models = convert_models(backend, f"{workspace}/predictor_build/results/{kernel_type}_{mark}.json")

# run models with given backend and return latency of testcase models
profiled_results = profile_models(backend, models, mode='predbuild', have_converted=True,
                                  save_name=f"profiled_{kernel_type}.json")
```

Note: for kernels related to `conv` or `dwconv`, our experiment results have shown that all kernels containing one `conv` layer have almost the same latency results, as `conv` layer has dominant latency. For example, `conv-bn-relu` has almost the same latency as `conv-block`. Same observation was found for `dwconv` related kernels. Therefore in nn-Meter, all `conv` related kernels shares the same kernel predictor, so does `dwconv` related kernels.

## Step 4: Initialize Kernel Latency Predictor

After preparing the training data, we construct a random forest regression model as the kernel latency predictor. Here is an example:

```python
import os
from nn_meter.builder import builder_config
from nn_meter.builder.kernel_predictor_builder import build_predictor_by_data
workspace = "/path/to/workspace/"
builder_config.init(workspace)

kernel_type = "conv-bn-relu"
mark = "test"
backend = "tflite_cpu"
error_threshold = 0.1

# extract training feature and target from profiled results
cfgs_path = os.path.join(workspace, "predictor_build", "results", f"{kernel_type}_{mark}.json")
lats_path = os.path.join(workspace, "predictor_build", "results", f"profiled_{kernel_type}.json")
kernel_data = (cfgs_path, lats_path)

# build latency predictor
predictor, acc10, error_configs = build_predictor_by_data(
    kernel_type, kernel_data, backend, error_threshold=error_threshold, mark=mark,
    save_path=os.path.join(workspace, "predictor_build", "results")
)
print(f'Iteration 0: acc10 {acc10}, error_configs number: {len(error_configs)}')
```

In the implementation, the tuple of `kernel_data` includes items of `cfgs_path` and `lats_path`, indicating the config information and latency information respectively. `cfgs_path` and `lats_path` accept both json file path string or dictionary of models. In addition, if the config information and latency information are in the same data holder, users could directly specify `kernel_data = cfgs_path`.

In the feature extraction part, the all configs are extracted as features. Besides, for kernels containing `conv`, `dwconv` or `fc` layer, flops and number of parameters are also extracted as features for latency prediction.

After feature extraction, nn-Meter build the predictor by method `build_predictor_by_data`. The regression predictor are implemented by `sklearn.ensemble.RandomForestRegressor`. All data are split into training and validation data as 8:2. We have evaluate the prediction performance by the Root Mean Square Error (RMSE) and the relative Root Mean SQuare Percentage Error (RMSPE), that are the standard metrics in regression. Besides, we report the $\pm 5\%$ and $\pm 10\%$ accuracy, that are the percentage of models with predicted latency within the corresponding error bound relative to the measured latency. Smaller `RMSE/RMSPE` and larger  $\pm 5\%$ and $\pm 10\%$ accuracy suggest better performance.

The output of `build_predictor_by_data` includes the predictor class, $\pm 10\%$ accuracy and training data items with larger error than `error_threshold`. The large error data are used for the next step.

nn-Meter also provides an API named `nn_meter.builder.build_initial_predictor_by_data` to integrate above three steps. Here is an example:

```python
from nn_meter.builder import builder_config
from nn_meter.builder.backends import connect_backend
from nn_meter.builder import build_initial_predictor_by_data
workspace = "/path/to/workspace/"
builder_config.init(workspace)

kernel_type = "conv-bn-relu"
backend = "tflite_cpu"
error_threshold = 0.1

predictor, data = build_initial_predictor_by_data(
    kernel_type, backend, init_sample_num=10, error_threshold=error_threshold
)
```

## Step 5: Adaptive Data Sampling

In the paper of nn-Meter, we have observe that the configuration of kernel size (`KERNEL_SIZE`), height and width (`HW`), input channel (`CIN`), and output channel (`COUT`) show the non-linearity pattern on our measured devices. Instead, `HW` and `COUT` exhibit the staircase pattern, in which Conv with two different `HW` / `COUT` may have the same latency. These non-linearities reflect the complexities in hardware optimizations.

Therefore, main idea to improve the predictor performance is to sample the most beneficial data from the kernel configuration space. It covers 1) the configuration range in CNN design, and 2) hardware-crucial configurations that reflect the hardware optimizaitons and can significantly impact the prediction accuracy.

We propose adaptive data sampling to generate fine-grained channel number sampling for data with large prediction errors. For each data, we fix all the other dimensions except the channel number $C_0$. we random sample $M$ data from $[0.4 \times C_0, 1.2 \times C_0]$. For example, for Conv with (HW=56, KERNEL_SIZE=3, STRIDES=1, CIN=24, COUT=64), we fix the HW, KERNEL_SIZE, STRIDES dimension, and sample $M$ new CIN, COUT from $[9, 28]$ and $[25, 76]$, respectively. The fine-grained sampling number is represented by parameter `finegrained_sample_num`.

The iterative process continues until the predictor accuracy meets user's requirements. In this part, we conduct the following steps for adaptive data sampling:

* Build a regression model by current sampled data;
* Locate data points in testset with large prediction error (prediction error > `large_error_threshold`, default=0.1);
* For each data point, we perform fine-grained data sampling to generate random data around the large error data;
* Collect fine-grained sampled data with previous data to build new predictor;
* Conduct next iteration.

Here is an example for adaptive data sampling:

```python
from nn_meter.builder import builder_config
from nn_meter.builder.backends import connect_backend
from nn_meter.builder import build_adaptive_predictor_by_data
workspace = "/path/to/workspace/"
builder_config.init(workspace)

kernel_type = "conv-bn-relu"
backend = "tflite_cpu"
error_threshold = 0.1

predictor, data = build_adaptive_predictor_by_data(
    kernel_type, kernel_data, backend, finegrained_sample_num=5
)
```
In the method `build_adaptive_predictor_by_data`, the parameter `kernel_data` indicates all training and testing data for current predictor training. The value of `kernel_data` could either be an instance of Dict generated by `build_initial_predictor_by_data` or `build_adaptive_predictor_by_data`, or be a instance of Tuple such as:

```python
config_json_file = [f'{workspace}/predictor_build/results/{kernel_type}_prior.json'] # Add all needed json files name in the list
latency_json_file = [f'{workspace}/predictor_build/results/profiled_{kernel_type}.json'] # Add all needed json files name in the list
kernel_data = (config_json_file, latency_json_file)
```

## End-to-end Demo

nn-Meter have wrapped the four main steps into one method named `nn_meter.builder.build_predictor_for_kernel`. There is an example to build latency predictor for `"conv-bn-relu"` kernel:

```python
# initialize builder config with workspace
from nn_meter.builder import builder_config
workspace = "/path/to/workspace/"
builder_config.init(workspace)

# build latency predictor for kernel
from nn_meter.builder import build_predictor_for_kernel
kernel_type = "conv-bn-relu"
backend = "tflite_cpu"

predictor, data = build_predictor_for_kernel(
    kernel_type, backend, init_sample_num=1000, finegrained_sample_num=10, iteration=5, error_threshold = 0.1
)
```

In the experiment of nn-Meter, we set default `init_sample_num` as 1000, `finegrained_sample_num` as 10, `iteration` as 5, and `error_threshold` as 0.1.

nn-Meter also provided a end-to-end method for users to build a series of general latency predictors, named `nn_meter.builder.build_latency_predictor`. This method will build predictors for all kernels in `<workspace-path>/configs/predictorbuild_config.yaml` according to their corresponding parameters. The parameters includes `INIT_SAMPLE_NUM`, `FINEGRAINED_SAMPLE_NUM`, `ITERATION`, and `ERROR_THRESHOLD`. Here is an example:

``` python
# initialize builder config with workspace
from nn_meter.builder import builder_config
workspace = "/path/to/workspace/"
builder_config.init(workspace)

# build latency predictor for kernel
from nn_meter.builder import build_latency_predictor
build_latency_predictor(backend="tflite_cpu")
```

# Kernel Data Format

## Structure of Kernel Data

In the process to build kernel latency predictor, a series of kernel data will be sampled, generated, and profiled to build the training dataset. One piece of complete kernel data consists of two parts, the configuration information and the profiled results (in default is the latency value). The configuration information and profiled results are dumped in json file to `<workspace-path>/predictor_build/results/<kernel_type>_<mark>.json` and `<workspace-path>/predictor_build/results/profiled_<kernel_type>.json`， respectively. In each data piece, `"model"` points to its directory to the path of this kernels' `Keras` model, `"shapes"` indicates the input shape of the tensor to test, and `"latency"` reports the profiled results after running `profile_models`. The ids of kernel data are randomly generated and consists of 6 capital letters.

This is a json dump of the configuration information of generated kernels, which we call it config json file:

```json
"conv-bn-relu": {
    "YB2F4N": {
        "model": "<workspace-path>/predictor_build/kernels/conv-bn-relu_prior_YB2F4N",
        "shapes": [
            [
                13,
                13,
                212
            ]
        ],
        "config": {
            "HW": 13,
            "CIN": 212,
            "COUT": 176,
            "KERNEL_SIZE": 1,
            "STRIDES": 1
        }
    }
}
```

After running and profiling the kernels, the `"latency"` attribute appears in `<workspace-path>/predictor_build/results/profiled_<kernel_type>.json`, which we call it latency json file:

```json
"conv-bn-relu": {
    "YB2F4N": {
        "latency": "37.53 +- 0.314"
    }
}
```
Note: If the parameter `DETAIL` is `TRUE` in `<workspace-path>/configs/predictorbuild_config.yaml`, the configuration information will also be dumped in `<workspace-path>/predictor_build/results/profiled_<kernel_type>.json`.

Config json file and latency json file formed the training data of kernel latency predictor. The kernel data should be defined as follows:

```python
config_json_file = [
    f'{workspace}/predictor_build/results/{kernel_type}_prior.json',
    f'{workspace}/predictor_build/results/{kernel_type}_finegrained1.json',
    f'{workspace}/predictor_build/results/{kernel_type}_finegrained2.json'
]
latency_json_file = [
    f'{workspace}/predictor_build/results/profiled_{kernel_type}.json'
]
kernel_data = (config_json_file, latency_json_file)
```

and called as follows:
```python
# build initial latency predictor by kernel_data
predictor, acc10, error_configs = build_predictor_by_data(
    kernel_type, kernel_data, backend, error_threshold=error_threshold, mark="prior",
    save_path=os.path.join(workspace, "predictor_build", "results")
)
```
or:
```python
# build adaptive latency predictor by kernel_data
predictor, data = build_adaptive_predictor_by_data(
    kernel_type, kernel_data, backend, finegrained_sample_num=5
)
```

## Convert Kernel Data to CSV

nn-Meter provides method to convert kernel data json files to CSV. Here is an example:

```python
from nn_meter.builder.kernel_predictor_builder.predictor_builder.utils import collect_kernel_data
from nn_meter.builder.kernel_predictor_builder.predictor_builder.extract_feature import get_feature_parser, get_data_by_profiled_results

kernel_type = 'conv-bn-relu'

# define kernel data
config_json_file = [
    f'{workspace}/predictor_build/results/{kernel_type}_prior.json',
    f'{workspace}/predictor_build/results/{kernel_type}_finegrained1.json',
    f'{workspace}/predictor_build/results/{kernel_type}_finegrained2.json'
]
latency_json_file = [
    f'{workspace}/predictor_build/results/profiled_{kernel_type}.json'
]
kernel_data = (config_json_file, latency_json_file)

# read kernel data and extract features
kernel_data = collect_kernel_data(kernel_data)
feature_parser = get_feature_parser(kernel_type) # define the feature to extract

data = get_data_by_profiled_results(kernel_type, feature_parser, kernel_data,
                                    save_path="path/to/csv/test.csv")
```


# Build Predictor for Customized Kernel

If users want to add new kernels to profile latency and build predictor, here are several steps to prepare and register new kernels.

## Prepare Customized Kernels

### Step 1: Prepare the Customized Kernel Class

nn-Meter provide API for users to customize their own kernel block. In nn-Meter, each kernel is implemented by inheriting a base class named `nn_meter.builder.nn_modules.BaseBlock`. The kernel block has a input parameter `config` to feed configuration params for the kernel. There are two attributes should be claimed, including `input_shape` and `input_tensor_shape`, as well as one method named `get_model()`. nn-Meter support both Tensorflow and PyTorch implementation for the kernel model. Users could switch the kernel implementation between Tensorflow and PyTorch by editing configuration `IMPLEMENT` in `<workspace-path>/configs/predictorbuild_config.yaml`. Here we use Tensorflow implementation as an example.

- `input_shape` defines the dimension of one model input shape without batch size. Generally, when the input shape is 3D, `input_shape` should be`[config["HW"], config["HW"], config["CIN"]]`, and when the input shape is 1D, `input_shape` should be`[config["CIN"]]`. 

- `input_tensor_shape` is a list defining all model inputs. In basic situation, `input_tensor_shape` should be `[input_shape]` if the kernel only has one input. If the kernel has more than one input, such as `addrelu` kernel, `input_tensor_shape` is `[input_shape, input_shape]`.

- `get_model` is the implementation of the kernel model and return a instance of `keras.Model` of the kernel.

Users could refer to the following example to learn how to write a kernel class.

``` python
import tensorflow.keras as keras
from nn_meter.builder.nn_modules import BaseBlock

class MyKernel(BaseBlock):
    ''' This kernel is built by Conv, BN, and Relu layer, which is the same as the builtin `conv-bn-relu` block.
    '''
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, cout, kernel_size, strides):
                super().__init__()
                self.conv = keras.layers.Conv2D(
                    cout,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="same"
                )
                self.bn = keras.layers.BatchNormalization()
                self.relu = keras.layers.ReLU()

            def call(self, inputs):
                x = self.conv(inputs)
                x = self.bn(x)
                x = self.relu(x)
                return x

        return Model(self.config["COUT"], self.config["KERNEL_SIZE"], self.config["STRIDES"])
```

### Step 2: Collect the Prior Data and Implement Sampling Code

Next, users should collect the prior data and implement the config sampler for customized kernel. In nn-Meter, config sampler of each kernel is implemented by inheriting a base class named `nn_meter.builder.kernel_predictor_builder.BaseConfigSampler`. The config sampler has two methods, including `prior_config_sampling` and `finegrained_config_sampling`. The output of both methods is a list of dicts, with each dict indicates a group of configuration.

- `prior_config_sampling(self, sample_num)`: utilize the prior data to define the configuration sampling from the prior distribution.

- `finegrained_config_sampling(self, sample_num, configs)`: for data in `configs`, perform fine-grained data sampling to generate random data around the large error data.

Here is an example:

``` python
import random
from nn_meter.builder.kernel_predictor_builder import BaseConfigSampler

class MySampler(BaseConfigSampler):
    ''' This sampler is for Conv related sampler. In `prior_config_sampling` method, all configs are sampled based on existing conv model. In
    `finegrained_config_sampling` method, only CIN and COUT are sampled around the configs in parameter `configs`.
    '''

    def prior_config_sampling(self, sample_num):
        new_hws = ...
        new_cins = ...
        new_couts = ...
        new_kernel_sizes = ...
        new_strides = ...
        ncfgs = []
        for hw, cin, cout, kernel_size, stride in zip(new_hws, new_cins, new_couts,
                                                      new_kernel_sizes, new_strides):
            c = {
                'HW': hw,
                'CIN': cin,
                'COUT': cout,
                'KERNEL_SIZE': kernel_size,
                'STRIDES': stride,
            }
            ncfgs.append(c)
        return ncfgs
    
    def finegrained_config_sampling(self, sample_num, configs):
        ncfgs = []
        for cfg in configs:
            cins = ...
            couts = ...
            for cin, cout in zip(cins, couts):
                c = {
                    'HW': cfg['HW'],
                    'CIN': cin,
                    'COUT': cout,
                    'KERNEL_SIZE': cfg['KERNEL_SIZE'],
                    'STRIDES': cfg['STRIDES'],
                }
                ncfgs.append(c)
        return ncfgs
```

Note: all sampled configuration value will be feed into the kernels by the input `config`. Users should follow the same notation in sampler and kernel class to transfer parameters.

### Step 3: Specify Kernel Feature for Training Predictor

Finally, users should specify the feature of kernel for training the kernel latency predictor. nn-Meter provide a base class named `nn_meter.builder.kernel_predictor_builder.BaseFeatureParser`. The feature parser has two needed attributes named `kernel_type` and `needed_config`, as well as two methods, including `get_feature_by_config(self, config_dict)` and `get_config_by_feature`.

- `kernel_type`: the builtin kernel type of the parser.

- `needed_config`: the list of all config variables. such as `["HW", "CIN", "KERNEL_SIZE", "STRIDES"]`.

- `get_feature_by_config(self, config_dict)`: convert the config_dict to feature list, new features based on configs is acceptable.

- `get_config_by_feature(self, feature)`: convert the feature to config_dict.The newly added feature should be removed.

Here is an example:

``` python
from nn_meter.builder.kernel_predictor_builder import BaseFeatureParser

class MyParser(BaseFeatureParser):
    ''' This parser utilized config "HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES", as well as the flops and parameter number as feature.
    '''
    def __init__(self, kernel_type):
        self.kernel_type = kernel_type
        self.needed_config = ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"]

    def get_feature_by_config(self, config_dict):
        feature = [config_dict[data] for data in self.needed_config]
        hw, cin, cout, kernel_size, stride = config_dict["HW"], config_dict["CIN"], config_dict["COUT"], \
            config_dict["KERNEL_SIZE"], config_dict["STRIDES"]
        param = cout * (kernel_size * kernel_size * cin + 1)
        flop = 2 * hw / stride * hw / stride * param

        flop /= 2e6
        param /= 1e6
        feature.extend([flop, param])
        return feature

    def get_config_by_feature(self, feature):
        # remove flops and params num feature from feature vector
        feature = feature[:-2]
        assert len(self.needed_config) == len(feature)
        config = {k: v for k, v in zip(self.needed_config, feature)}
        return config
```

## Register kernel to nn-Meter

### Step 1: Create a Package for the Customized Kernel

nn-Meter requires users to gather all code of kernel in a package with a fixed location. A folder will be treated as a package with a `__init__.py` file added. Here is a demo of folder structure:

``` text
./customized_kernel/
├── __init__.py
├── kernel_script.py
├── config_sampler.py
└── feature_parser.py
```

The interface of customized kernel class, named `MyKernel`, are stored in `./customized_kernel/kernel_script.py`, and customized sampler `MySampler` in `./customized_kernel/config_sampler.py`, as well as feature parser `MyParser` in `./customized_kernel/feature_parser.py`, respectively.

### Step 2: Prepare Meta File

Create a yaml file with following keys as meta file:

- `builtin_name`: builtin name used in nn-Meter configuration file to call the customized kernel, such as `"mykernel"`.

- `implement`: the implementation type of the customized kernel, chosen from ["tensorflow", "torch"].

- `package_location`: the absolute path of the package folder.

- `class_module`: the module of the kernel class, in this example is `kernel_script`, representing `kernel_script.py`.

- `class_name`: the kernel class name, in this example is `MyKernel`.

- `sampler_module`: the module of the kernel sampler, in this example is `config_sampler`, representing `config_sampler.py`.

- `sampler_name`: the kernel sampler name, in this example is `MySampler`.

- `parser_module`: the module of the kernel feature parser, in this example is `feature_parser`, representing `feature_parser.py`.

- `parser_name`: the kernel parser name, in this example is `MyParser`.

Following is an example of the yaml file:

```yaml
builtin_name: mykernel
implement: tensorflow
package_location: /home/{USERNAME}/working/kernel_package
class_module: kernel_script
class_name: MyKernel
sampler_module: config_sampler
sampler_name: MySampler
parser_module: feature_parser
parser_name: MyParser
```

Note: Different with registering [operator and test case](./test_fusion_rules.md#build-customized-test-cases), the registration of customized kernel doesn't support the same name with different implementation (i.e., tensorflow or torch). This is because except the kernel class, there are also parts of config sampler and feature parser to define a customized kernel, which could have a difference between different implementation. If you want to register the same kernel with different implementation, you should set different builtin names to distinguish them, such as "mykernel_tf" and "mykernel_torch".

### Step 3: Register Customized Kernel into nn-Meter

Run the following command to register customized kernel into nn-Meter:

``` bash
nn-meter register --kernel path/to/meta/file
```

If the registration success, nn-Meter will show:

``` text
(nn-Meter) Successfully register kernel: mykernel
```

When registering, nn-Meter will test whether the module can be imported first. If the registration success is not successful, please check the package according to the error information.

After backend registration, users can view all kernels by running:
``` bash
nn-meter --list-kernels
```
```text
(nn-Meter) Supported kernels: ('*' indicates customized kernels)
(nn-Meter) [Kernel] conv-bn-relu
(nn-Meter) [Kernel] conv-bn-relu6
(nn-Meter) [Kernel] conv-bn
(nn-Meter) [Kernel] conv-relu
(nn-Meter) [Kernel] conv-relu6
(nn-Meter) [Kernel] conv-hswish
(nn-Meter) [Kernel] conv-block
(nn-Meter) [Kernel] conv-bn-hswish
(nn-Meter) [Kernel] dwconv-bn
(nn-Meter) [Kernel] dwconv-relu
(nn-Meter) [Kernel] dwconv-relu6
(nn-Meter) [Kernel] dwconv-bn-relu
(nn-Meter) [Kernel] dwconv-bn-relu6
(nn-Meter) [Kernel] dwconv-block
(nn-Meter) [Kernel] dwconv-bn-hswish
(nn-Meter) [Kernel] maxpool
(nn-Meter) [Kernel] avgpool
(nn-Meter) [Kernel] fc
(nn-Meter) [Kernel] concat
(nn-Meter) [Kernel] split
(nn-Meter) [Kernel] channelshuffle
(nn-Meter) [Kernel] se
(nn-Meter) [Kernel] global-avgpool
(nn-Meter) [Kernel] bnrelu
(nn-Meter) [Kernel] bn
(nn-Meter) [Kernel] hswish
(nn-Meter) [Kernel] relu
(nn-Meter) [Kernel] addrelu
(nn-Meter) [Kernel] add
(nn-Meter) [Kernel] * mykernel
```

Note: the package of customized kernel must be retained in a fixed path as registered one. Otherwise may cause error when calling the registered module.

## Use the Customized Kernel in Experiment

After registration, users could build latency predictor for the customized kernel:

```python
# initialize builder config with workspace
from nn_meter.builder import builder_config
builder_config.init("path/to/workspace/folder") 

# build latency predictor for customized kernel
from nn_meter.builder import build_predictor_for_kernel
kernel_type = "mykernel"
backend = "tflite_cpu"

predictor, data = build_predictor_for_kernel(
    kernel_type, backend, init_sample_num = 1000, finegrained_sample_num = 10, iteration = 5, error_threshold = 0.1
)
```

## Manage the Registered Kernel

Users could unregister the kernel by calling its name in command:

``` bash
nn-meter unregister --kernel mykernel
```
``` text
(nn-Meter) Successfully unregister mykernel.
```

After unregister the kernel, "mykernel" will be removed from the backend list.
