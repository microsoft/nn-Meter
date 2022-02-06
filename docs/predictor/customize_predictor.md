# Customize Latency Predictor for Users' Device

nn-Meter support customized predictor for latency prediction. To utilize customized predictor, users should provide several kernel latency predictors and a fusion rule json file. Users could use [nn-Meter builder](../builder/overview.md) to build their own latency predictors.

After preparing kernel latency predictors and fusion rule following guidance [here](../builder/overview.md), users should register the predictor to nn-Meter for reuse. First of all, put all kernel latency predictors and the fusion rule json file into one folder in a fixed location. The kernel latency predictors should be named by the kernel name, such as `"conv-bn-relu.pkl"`. The fusion rule json file should be named as `"fusion_rules.json"`.

### Step 1: Prepare Meta File

Create a yaml file with following keys as meta file:

- `name`: builtin name used in nn-Meter configuration file to call the customized predictor, such as `"my_predictor"`.

- `version`: the version of the customized predictor.

- `category`: the category of the backend platform of the predictor, such as `"cpu"` or `"gpu"`.

- `package_location`: the absolute path of the folder containing all predictors and fusion rule file.

- `kernel_predictors`: list all kernel latency predictors. Note that the name of predictor `.pkl` file should be as the same as the listed one.

Following is an example of the yaml file:

```yaml
name: my_predictor
version: 1.0
category: cpu
package_location: /home/USERNAME/working/customized_predictor
kernel_predictors:
    - conv-bn-relu
    - dwconv-bn-relu
    - fc
    - global-avgpool
    - hswish
    - relu
    - se
    - split
    - add
    - addrelu
    - maxpool
    - avgpool
    - bn
    - bnrelu
    - channelshuffle
    - concat
```

### Step 2: Register Customized Predictor into nn-Meter

Run the following command to register customized predictor into nn-Meter:

``` bash
nn-meter register --predictor path/to/meta/file
```
If the registration success, nn-Meter will show:
``` text
(nn-Meter) Successfully register predictor my_predictor
```

After predictor registration, users can view all predictors by running:
``` bash
nn-meter --list-predictors
```
```text
(nn-Meter) Supported latency predictors:
(nn-Meter) [Predictor] cortexA76cpu_tflite21: version=1.0
(nn-Meter) [Predictor] adreno640gpu_tflite21: version=1.0
(nn-Meter) [Predictor] adreno630gpu_tflite21: version=1.0
(nn-Meter) [Predictor] myriadvpu_openvino2019r2: version=1.0
(nn-Meter) [Predictor] my_predictor: version=1.0
```

Note: the folder of customized predictor must be retained in a fixed path as registered one. Otherwise may cause error when calling the registered module.

## Use the Customized Backend in Experiment

After registration, users could get access to the customized predictor by the same way as the builtin predictors. Following [here](usage.md) to get all usages.


## Manage the Registered Predcitor

Users can view all builtin and registered predictors by running:
``` bash
nn-meter --list-predictors
```
```text
(nn-Meter) Supported latency predictors:
(nn-Meter) [Predictor] cortexA76cpu_tflite21: version=1.0
(nn-Meter) [Predictor] adreno640gpu_tflite21: version=1.0
(nn-Meter) [Predictor] adreno630gpu_tflite21: version=1.0
(nn-Meter) [Predictor] myriadvpu_openvino2019r2: version=1.0
(nn-Meter) [Predictor] my_predictor: version=1.0
```

Besides, users could unregister the predictor by calling its name in command:

``` bash
nn-meter unregister --predictor my_predictor
```
``` text
(nn-Meter) Successfully unregister my_predictor.
```

After unregister the predictor, "my_predictor" will be removed from the predictor list:

``` bash
nn-meter --list-predictors
```
``` text
(nn-Meter) Supported latency predictors:
(nn-Meter) [Predictor] cortexA76cpu_tflite21: version=1.0
(nn-Meter) [Predictor] adreno640gpu_tflite21: version=1.0
(nn-Meter) [Predictor] adreno630gpu_tflite21: version=1.0
(nn-Meter) [Predictor] myriadvpu_openvino2019r2: version=1.0
```