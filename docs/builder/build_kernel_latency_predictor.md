# Build Kernel Latency Predictor

## Step1: Config Sampling From Prior Distribution

To learn the relationship between configurations and latency, we need to generate a training set (i.e., variously configured kernels and the latencies) for regression. While it's unfeasible to sample and measure all the configurations for all kernels, a direct method is random sampling.

The first step is sampling configuration values from the prior distribution, which is inferred from the existing models. Based on our kernel model, there are generally 6 configuration values, including height and width (`"HW"`), input channel (`"CIN"`), output channel (`"COUT"`), kernel size (`"KERNEL_SIZE"`), strides (`"STRIDES"`), and kernel size for pooling layer (`"POOL_STRIDES"`). We sampling the configuration based on the prior distribution and adapt the value to common valid values. That is, height and weight are verified to value from `[1, 3, 7, 14, 28, 56, 112, 224]`, kernel size to `[1, 3, 5, 7]`, strides to `[1, 2, 4]`, and kernel size for pooling layer to `[2, 3]`. We stored the prior knowledge of existing models as csv files in `nn_meter/builder/kernel_predictor_builder/data_sampler/prior_config_lib/`.

## Step 2: Generate and Profile Kernel Model by Configs

The second step is generating and profiling kernel model by configurations. Currently, the kernel blocks and corresponding configurations supported by nn-Meter include:

(conv related kernels)

- `"conv_bn_relu"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `STRIDES`
- `"conv_bn_relu6"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `STRIDES`
- `"conv_bn"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `STRIDES`
- `"conv_relu"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `STRIDES`
- `"conv_relu6"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `STRIDES`
- `"conv_hswish"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `STRIDES`
- `"conv_block"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `STRIDES`
- `"conv_bn_hswish"`: `HW`, `CIN`, `COUT`, `KERNEL_SIZE`, `STRIDES`

(dwconv related kernels)
- `"dwconv_bn"`: `HW`, `CIN`, `KERNEL_SIZE`, `STRIDES`
- `"dwconv_relu"`: `HW`, `CIN`, `KERNEL_SIZE`, `STRIDES`
- `"dwconv_relu6"`: `HW`, `CIN`, `KERNEL_SIZE`, `STRIDES`
- `"dwconv_bn_relu"`: `HW`, `CIN`, `KERNEL_SIZE`, `STRIDES`
- `"dwconv_bn_relu6"`: `HW`, `CIN`, `KERNEL_SIZE`, `STRIDES`
- `"dwconv_block"`: `HW`, `CIN`, `KERNEL_SIZE`, `STRIDES`
- `"dwconv_bn_hswish"`: `HW`, `CIN`, `KERNEL_SIZE`, `STRIDES`

(other kernels)

- `"maxpool_block"`: `HW`, `CIN`, `KERNEL_SIZE`, `POOL_STRIDES`
- `"avgpool_block"`: `HW`, `CIN`, `KERNEL_SIZE`, `POOL_STRIDES`
- `"fc_block"`: `CIN`, `COUT`
- `"concat_block"`: `HW`, `CIN1`, `CIN2`, `CIN3`, `CIN4`
- `"split_block"`: `HW`, `CIN`
- `"channel_shuffle"`: `HW`, `CIN`
- `"se_block"`: `HW`, `CIN`
- `"globalavgpool_block"`: `HW`, `CIN`
- `"bn_relu"`: `HW`, `CIN`
- `"bn_block"`: `HW`, `CIN`
- `"hswish_block"`: `HW`, `CIN`
- `"relu_block"`: `HW`, `CIN`
- `"add_relu"`: `HW`, `CIN`
- `"add_block"`: `HW`, `CIN`


The first and second step are implemented by `nn_meter.builder.nn_meter_builder.sample_and_profile_kernel_data`. Here is an example:

``` python
from nn_meter.builder.nn_meter_builder import sample_and_profile_kernel_data
kernel_type = "conv_bn_relu"
backend = "tflite_cpu"

# init predictor builder with prior data sampler
kernel_data = sample_and_profile_kernel_data(kernel_type, init_sample_num = 1000, backend, sampling_mode='prior', mark='prior')
```

The generated models are saved in `<workspace-path>/predictor_build/models`, and the configuration information and profiled results are dumped in json file to `<workspace-path>/predictor_build/results/<kernel_type>.json` and `<workspace-path>/predictor_build/results/profiled_<kernel_type>.json`.

The method `sample_and_profile_kernel_data` is composed by three main steps, `generate_config_sample`, `convert_models`, `profile_models`. Here is an example as a decomposition of `sample_and_profile_kernel_data`. Users could choose the decomposed interfaces if needed.
``` python
from nn_meter.builder.kernel_predictor_builder import generate_config_sample
from nn_meter.builder import convert_models
# sample configs for kernel and generate models
models = generate_config_sample(kernel_type, sample_num, mark=mark, 
                                    sampling_mode=sampling_mode, configs=configs)

# connect to backend, run models and get latency
backend = connect_backend(backend_name="tflite-cpu")

# convert the model to the needed format by backend, in order to increase efficiency when profiling on device.
models = convert_models(backend, saved_name, broken_point_mode=True)

# run models with given backend and return latency of testcase models
profiled_results = profile_models(backend, models, mode='predbuild', save_name="xxx.json", have_converted=True)
```

Note: for kernels related to conv and dwconv, our experiment results have shown that all kernels containing one conv layer or one dwconv layer have almost the same latency results. Thus in nn-Meter, all kernels containing one conv or dwconv layer shares the same kernel predictor.

## Step 3: Initialize Kernel Latency Predictor

After preparing the training data, we construct a random forest regression model as the kernel latency predictor. Here is an example:

```python
from nn_meter.builder.kernel_predictor_builder import build_predictor_by_data

kernel_type = "conv_bn_relu"
backend = "tflite_cpu"
error_threshold = 0.1

# extract training feature and target from profiled results
cfgs_path = os.path.join("<workspace-path>", "predictor_build", "results", "conv_bn_relu.json")
lats_path = os.path.join("<workspace-path>", "predictor_build", "results", "profiled_conv_bn_relu.json")
kernel_data = (cfgs_path, lats_path)

# build latency predictor
predictor, acc10, error_configs = build_predictor_by_data(
    kernel_type, kernel_data, backend, error_threshold=error_threshold, mark="prior",
    save_path=os.path.join("<workspace-path>", "predictor_build", "results")
)
logging.info(f'Iteration 0: acc10 {acc10}, error_configs number: {len(error_configs)}')    
```

In the implementation, the tuple of `kernel_data` includes items of `cfgs_path` and `lats_path`, indicating the config information and latency information respectively. `cfgs_path` and `lats_path` accept both json file path string or dictionary of models. In addition, if the config information and latency information are in the same data holder, users could directly specify `kernel_data = cfgs_path`.

In the feature extraction part, the all configs are extracted as features. Besides, for kernels containing `conv`, `dwconv` or `fc` layer, flops and number of parameters are also extracted as features for latency prediction.

After feature extraction, nn-Meter build the predictor by method `build_predictor_by_data`. The regression predictor are implemented by `sklearn.ensemble.RandomForestRegressor`. All data are split into training and validation data as 8:2. We have evaluate the prediction performance by the Root Mean Square Error (RMSE) and the relative Root Mean SQuare Percentage Error (RMSPE), that are the standard metrics in regression. Besides, we report the $\pm 5\%$ and $\pm 10\%$ accuracy, that are the percentage of models with predicted latency within the corresponding error bound relative to the measured latency. Smaller `RMSE/RMSPE` and larger  $\pm 5\%$ and $\pm 10\%$ accuracy suggest better performance.

The output of `build_predictor_by_data` includes the predictor class, $\pm 10\%$ accuracy and training data items with larger error than `error_threshold`. The large error data are used for the next step.

## Step 4: Adaptive Data Sampling

In the paper of nn-Meter, we have observe that the configuration of kernel size (KERNEL_SIZE), height and width (HW), input channel (CIN), and output channel (COUT) show the noe-linearity pattern on our measured devices. Instead, HW and COUT exhibit the staircase pattern, in which Conv with two different HW/COUT may have the same latency. These non-linearities reflect the complexities in hardware optimizations.

Therefore, main idea to improve the predictor performance is to sample the most beneficial data from the kernel configuration space. It covers 1) the configuration range in CNN design, and 2) hardware-crucial configurations that reflect the hardware optimizaitons and can significantly impact the prediction accuracy.

We propose adaptive data sampling to generate fine-grained channel number sampling for data with large prediction errors. For each data, we fix all the other dimensions except the channel number $C_0$. we random sample $M$ data from $[0.4 \times C_0, 1.2 \times C_0]$. For example, for Conv with (HW=56, KERNEL_SIZE=3, STRIDES=1, CIN=24, COUT=64), we fix the HW, KERNEL_SIZE, STRIDES dimension, and sample $M$ new CIN, COUT from $[9, 28]$ and $[25, 76]$, respectively. The fine-grained sampling number is represented by parameter `finegrained_sample_num`.

The iterative process continues until the predictor accuracy meets user's requirements. In this part, we conduct the following steps:

* Build a regression model by current sampled data;
* Locate data points in testset with large prediction error (prediction error >`large_error_threshold`, default=0.1);
* For each data point, we perform fine-grained data sampling to generate random data around the large error data;
* Collect fine-grained sampled data with previous data to build new predictor;
* Conduct next iteration.

Here is an example for adaptive data sampling:
```python
for i in range(1, iteration):
    # finegrained sampling and profiling for large error data
    new_kernel_data = sample_and_profile_kernel_data(
        kernel_type, finegrained_sample_num, backend,
        sampling_mode = 'finegrained', configs=error_configs, mark=f'finegrained{i}'
    )

    # merge finegrained data with previous data and build new regression model
    kernel_data = merge_prev_info(new_info=new_kernel_data, prev_info=kernel_data)
    predictor, acc10, error_configs = build_predictor_by_data(
        kernel_type, kernel_data, backend, error_threshold=error_threshold, mark="prior",
        save_path=os.path.join("<workspace-path>", "predictor_build", "results")
        )
    logging.keyinfo(f'Iteration {i}: acc10 {acc10}, error_configs number: {len(error_configs)}')
```

## End-to-end Demo

nn-Meter have wrapped the four main steps into one method named `nn_meter.builder.build_predictor_for_kernel`. There is an example to build latency predictor for `"conv_bn_relu"` kernel:

```python
# initialize builder config with workspace
from nn_meter.builder import builder_config
builder_config.init("path/to/workspace/folder") 

# build latency predictor for kernel
from nn_meter.builder import build_predictor_for_kernel
kernel_type = "conv_bn_relu"
backend = "tflite_cpu"

predictor, data = build_predictor_for_kernel(
    kernel_type, backend, init_sample_num = 1000, finegrained_sample_num = 10, iteration = 5, error_threshold = 0.1
)
```

In the experiment of nn-Meter, we set `init_sample_num` as 1000, `finegrained_sample_num` as 10, `iteration` as 5, and `error_threshold` as 0.1.

nn-Meter also provided a end-to-end method for users to build a series of general latency predictors, named `nn_meter.builder.build_latency_predictor()`. This method will build predictors for all kernels in `<workspace-path>/configs/predictorbuild_config.yaml` according to their corresponding parameters. The parameters includes `INIT_SAMPLE_NUM`, `FINEGRAINED_SAMPLE_NUM`, `ITERATION`, and `ERROR_THRESHOLD`. Here is an example:

``` python
# initialize builder config with workspace
from nn_meter.builder import builder_config
builder_config.init("path/to/workspace/folder") # initialize builder config with workspace

# build latency predictor for kernel
from nn_meter.builder import build_latency_predictor
build_latency_predictor(backend="tflite_cpu")
```

# Build predictor for customized kernel

If users want to add new kernels to profile latency and build predictor, here are several steps to prepare and register new kernels.

## Prepare Customized Kernels

### Step 1: Prepare the Customized Kernel Class

nn-Meter provide API for users to customize their own kernel block. In nn-Meter, each kernel is implemented by inheriting a base class named `nn_meter.builder.nn_generator.BaseBlock`. The kernel block has a input parameter `config` to feed configuration params for the kernel. There are two attributes should be claimed, including `input_shape` and `input_tensor_shape`, as well as one method named `get_model()`. nn-Meter support both tensorflow and torch implementation for the kernel model. Here we use tensorflow implementation as an example. 

- `input_shape` defines the dimension of one model input shape without batch size. Generally, when the input shape is 3D, `input_shape` should be`[config["HW"], config["HW"], config["CIN"]]`, and when the input shape is 1D, `input_shape` should be`[config["CIN"]]`. 

- `input_tensor_shape` is a list defining all model inputs. In basic situation, `input_tensor_shape` should be `[input_shape]` if the kernel only has one input. If the kernel has more than one input, such as `add_relu` kernel, `input_tensor_shape` is `[input_shape, input_shape]`.

- `get_model` is the implementation of the kernel model and return a instance of `keras.Model` of the kernel.

Users could refer to the following example to learn how to write a kernel class.

``` python
import tensorflow.keras as keras
from nn_meter.builder.nn_generator import BaseBlock

class MyKernel(BaseBlock):
    ''' This kernel is built by Conv, BN, and Relu layer, which is the same as the builtin `conv_bn_relu` block.
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
    `finegrained_config_sampling` method, only cin and cout are sampled around the configs in parameter `configs`.
    '''

    def prior_config_sampling(self, sample_num):
        new_hws = ...
        new_cins = ...
        new_couts = ...
        new_kernel_sizes = ...
        new_strides = ...
        for hw, cin, cout, kernel_size, stride in zip(new_hws, new_cins, new_couts, new_kernel_sizes, new_strides):
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
    ''' This parser utilized config "HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES", as well as the flops and parameter number as feature, 
    which is the same parser for Conv, Dwconv and FC related kernel.
    '''
    def __init__(self, kernel_type):
        self.kernel_type = kernel_type
        self.needed_config = ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"]

    def get_feature_by_config(self, config_dict):
        feature = [config_dict[data] for data in self.needed_config]
        hw, cin, cout, kernel_size, stride = config_dict["HW"], config_dict["CIN"], config_dict["COUT"], \
            config_dict["KERNEL_SIZE"], config_dict["STRIDES"]
        param = cout * (kernel_size * kernel_size + 1)
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
package_location: /home/USERNAME/working/tftest/kernel_package
class_module: kernel_script
class_name: MyKernel
sampler_module: config_sampler
sampler_name: MySampler
parser_module: feature_parser
parser_name: MyParser
```

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
(nn-Meter) [Kernel] conv_bn_relu
(nn-Meter) [Kernel] conv_bn_relu6
(nn-Meter) [Kernel] conv_bn
(nn-Meter) [Kernel] conv_relu
(nn-Meter) [Kernel] conv_relu6
(nn-Meter) [Kernel] conv_hswish
(nn-Meter) [Kernel] conv_block
(nn-Meter) [Kernel] conv_bn_hswish
(nn-Meter) [Kernel] dwconv_bn
(nn-Meter) [Kernel] dwconv_relu
(nn-Meter) [Kernel] dwconv_relu6
(nn-Meter) [Kernel] dwconv_bn_relu
(nn-Meter) [Kernel] dwconv_bn_relu6
(nn-Meter) [Kernel] dwconv_block
(nn-Meter) [Kernel] dwconv_bn_hswish
(nn-Meter) [Kernel] maxpool_block
(nn-Meter) [Kernel] avgpool_block
(nn-Meter) [Kernel] fc_block
(nn-Meter) [Kernel] concat_block
(nn-Meter) [Kernel] split_block
(nn-Meter) [Kernel] channel_shuffle
(nn-Meter) [Kernel] se_block
(nn-Meter) [Kernel] globalavgpool_block
(nn-Meter) [Kernel] bn_relu
(nn-Meter) [Kernel] bn_block
(nn-Meter) [Kernel] hswish_block
(nn-Meter) [Kernel] relu_block
(nn-Meter) [Kernel] add_relu
(nn-Meter) [Kernel] add_block
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

