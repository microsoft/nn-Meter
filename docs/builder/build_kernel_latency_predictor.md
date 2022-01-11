# Build Kernel Latency Predictor

## Step1: Config Sampling From Prior Distribution

To learn the relationship between configurations and latency, we need to generate a training set (i.e., variously configured kernels and the latencies) for regression. While it's unfeasible to sample and measure all the configurations for all kernels, a direct method is random sampling.

The first step is sampling configuration values from the prior distribution, which is inferred from the existing models. Based on our kernel model, there are generally 6 configuration values, including height and width (`"HW"`), input channel (`"CIN"`), output channel (`"COUT"`), kernel size (`"KERNEL_SIZE"`), strides (`"STRIDES"`), and kernel size for pooling layer (`"POOL_STRIDES"`). We sampling the configuration based on the prior distribution and adapt the value to common valid values. That is, height and weight are verified to value from `[1, 3, 7, 14, 28, 56, 112, 224]`, kernel size to `[1, 3, 5, 7]`, strides to `[1, 2, 4]`, and kernel size for pooling layer to `[2, 3]`. We stored the prior knowledge of existing models as csv files in `nn_meter/builder/kernel_predictor_builder/data_sampler/prior_config_lib/`.

## Step 2: Generate and Profile Kernel Model by Configs

The second step is generating and profiling kernel model by configurations. Currently, the kernel blocks and corresponding configurations supported by nn-Meter include:

```python
config_for_kernel = {
    # conv related kernels
    "conv_bn_relu":         ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv_bn_relu6":        ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv_bn":              ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv_relu":            ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv_relu6":           ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv_hswish":          ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv_block":           ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "conv_bn_hswish":       ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    # dwconv related kernels
    "dwconv_bn":            ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
    "dwconv_relu":          ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
    "dwconv_relu6":         ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
    "dwconv_bn_relu":       ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
    "dwconv_bn_relu6":      ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
    "dwconv_block":         ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
    "dwconv_bn_hswish":     ["HW", "CIN", "KERNEL_SIZE", "STRIDES"],
    # other kernels
    "maxpool_block":        ["HW", "CIN", "KERNEL_SIZE", "POOL_STRIDES"],
    "avgpool_block":        ["HW", "CIN", "KERNEL_SIZE", "POOL_STRIDES"],
    "fc_block":             ["CIN", "COUT"],
    "concat_block":         ["HW", "CIN1", "CIN2", "CIN3", "CIN4"],
    "split_block":          ["HW", "CIN"],
    "channel_shuffle":      ["HW", "CIN"],
    "se_block":             ["HW", "CIN"],
    "globalavgpool_block": ["HW", "CIN"],
    "bn_relu":              ["HW", "CIN"],
    "bn_block":             ["HW", "CIN"],
    "hswish_block":         ["HW", "CIN"],
    "relu_block":           ["HW", "CIN"],
    "add_relu":             ["HW", "CIN"],
    "add_block":            ["HW", "CIN"], 
}
```

The first and second step are implemented by `nn_meter.builder.nn_meter_builder.sample_and_profile_kernel_data`. Here is an example:

``` python
from nn_meter.builder.nn_meter_builder import sample_and_profile_kernel_data
kernel_type = "conv_bn_relu"
backend = "tflite_cpu"

# init predictor builder with prior data sampler
kernel_data = sample_and_profile_kernel_data(kernel_type, init_sample_num = 1000, backend, sampling_mode='prior', mark='prior')
```

The generated models are saved in `<workspace-path>/predictor_build/models`, and the configuration information and profiled results are dumped in json file to `<workspace-path>/predictor_build/results/<kernel_type>.json` and `<workspace-path>/predictor_build/results/profiled_<kernel_type>.json`.

Note: for kernels related to conv and dwconv, our experiment results have shown that all kernels containing one conv layer or one dwconv layer have almost the same latency results. Thus in nn-Meter, all kernels containing one conv or dwconv layer shares the same kernel predictor.

## Step 3: Initialize Kernel Latency Predictor

After preparing the training data, we construct a random forest regression model as the kernel latency predictor. Here is an example:

```python
from nn_meter.builder.kernel_predictor_builder import build_predictor_by_data, get_data_by_profiled_results

kernel_type = "conv_bn_relu"
backend = "tflite_cpu"
error_threshold = 0.1

# extract training feature and target from profiled results
configs_file = os.join("<workspace-path>", "predictor_build", "results", "conv_bn_relu.json")
latency_file = os.join("<workspace-path>", "predictor_build", "results", "profiled_conv_bn_relu.json")
data = get_data_by_profiled_results(kernel_type, configs_file, latency_file)

# build latency predictor
predictor, acc10, error_configs = build_predictor_by_data(kernel_type, data, backend, error_threshold=error_threshold)
logging.info(f'Iteration 0: acc10 {acc10}, error_configs number: {len(error_configs)}')    
```

In the implementation, the method `get_data_by_profiled_results` accept both json file path string or dictionary of models. In `get_data_by_profiled_results`, the parameter `cfgs_path` and `lats_path` indicate the config information and latency information respectively. In addition, if the config information and latency information are in the same data holder, users could only specify `cfgs_path` and leave the `lats_path` as `None`.

In the feature extraction part, the all configs are extracted as features. Besides, for kernels containing conv, dwconv or fc layer, flops and number of parameters are also extracted as features for latency prediction.

After feature extraction, nn-Meter build the predictor by method `build_predictor_by_data`. The regression predictor are implemented by `sklearn.ensemble.RandomForestRegressor`. All data are split into training and validation data as 8:2. We have evaluate the prediction performance by the Root Mean Square Error (RMSE) and the relative Root Mean SQuare Percentage Error (RMSPE), that are the standard metrics in regression. Besides, we report the $\pm 5\%$ and $\pm 10\%$ accuracy, that are the percentage of models with predicted latency within the corresponding error bound relative to the measured latency. Smaller RMSE/RMSPE and larger  $\pm 5\%$ and $\pm 10\%$ accuracy suggest better performance.

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
    data = get_data_by_profiled_results(kernel_type, kernel_data)
    predictor, acc10, error_configs = build_predictor_by_data(kernel_type, data, backend, error_threshold=error_threshold)
    logging.keyinfo(f'Iteration {i}: acc10 {acc10}, error_configs number: {len(error_configs)}')
```

## End-to-end Demo

nn-Meter have wrapped the four main steps into one method `nn_meter.builder.build_predictor_for_kernel`. There is an example to build latency predictor for `"conv_bn_relu"` kernel:

```python
# initialize builder config with workspace
from nn_meter.builder.utils import builder_config
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


# Build predictor for customized kernel
TODO registration: kernels in blocks.py
if user has a new kernel, he/she needs to add: 1) tf kernel code generation in `generator/networks`. 2) collect the prior data and implement sampling code.
