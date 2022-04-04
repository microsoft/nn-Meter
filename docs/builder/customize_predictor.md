# Use Customized Predictor for Latency Prediction

nn-Meter supports customized latency predictors, which can be built on users' devices. To utilize customized predictor in nn-Meter, users should provide all the necessary kernel latency predictors and a fusion rule json file. Users could use [nn-Meter builder](../builder/overview.md) to build their own latency predictors.

After preparing kernel latency predictors and fusion rules following guidance [here](../builder/overview.md), users could register the predictor to nn-Meter for reuse.

### Step 1: Collect predictors and fusion rules

After preparing kernel latency predictors and fusion rules, there will be a folder containing all kernel predictors in `<workspace-path>/predictor_build/results/predictors/`, and a json file containing the fusion rules in `<workspace-path>/fusion_rule_test/results/detected_fusion_rule.json`. The first step is to collect all kernel latency predictors and the fusion rule json file into one folder in a predefined location. The kernel latency predictors should be named by the kernel name with the training mark such as `"prior"` or `"finegrained"` removed. The fusion rule json file should be named as `"fusion_rules.json"`. Here is an example of the folder:

``` text
/home/{USERNAME}/working/customized_predictor
├── fusion_rules.json
├── add.pkl
├── addrelu.pkl
├── avgpool.pkl
├── bn.pkl
├── bnrelu.pkl
├── channelshuffle.pkl
├── concat.pkl
├── conv-bn-relu.pkl
├── dwconv-bn-relu.pkl
├── fc.pkl
├── global-avgpool.pkl
├── hswish.pkl
├── maxpool.pkl
├── meta.yaml
├── relu.pkl
├── se.pkl
└── split.pkl
```

### Step 2: Prepare Meta File

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
package_location: /home/{USERNAME}/working/customized_predictor
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

### Step 3: Register Customized Predictor into nn-Meter

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

## Use the Customized Latency Predictor

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
