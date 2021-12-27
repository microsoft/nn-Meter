# Build the Rule Tester

A rule tester creates a series of models (what we call "test cases" in nn-Meter), runs the models on a given device, profiles the model to get its latency, and finally, detects the fusion rules for every pair of ops. To build a rule tester, there are four steps to implement the rule detection.

## Step 1. Prepare Backends and Create Workspace

The first step to run rule_tester is to prepare backends and create workspace. Users could follow guidance [here](./backend.md) for this step.

After creating the workspace, a yaml file named `ruletest_config.yaml` will be placed in `<workspace-path>/configs/`. The ruletest configs includes:

- `HW`: Default input shape of all test cases except those requiring 1d tensor input. Default value is `28`.
- `CIN`: Default input channel of all test cases. Default value is `16`.
- `SHAPE_1D`: Default input shape of all testcases that need 1d tensor input. E.g., fully connected layer. Default value is `428`.
- `COUT`: Default output channel (filter size). Default value is `256`.
- `KERNEL_SIZE`: Default kernel size. Default value is `3`.
- `PADDING`: Default padding type. Default value is `"same"`.
- `STRIDES`: Default strides size. Default value is `1`.
- `ENABLED`: The test cases that will be enabled. Currently we implement three kinds of rules, `BasicFusion` (code name `BF`), `MultipleOutNodes` (code name `MON`), `ReadyTensor` (code name `RT`). Among them, `BasicFusion` is the most important one, which will detect whether a pair of op can be fused. Default value is `['BF', 'MON', 'RT']`,
- `PARAMS`: The parameters for each test case. For example, here `eps` define the alpha in formula of [step 4](#step-4-detect-fusion-rule) to decide whether two ops can be fused for test cases of BasicFusion. Default value is:
    ```yaml
    BF:
        eps: 0.5
    ```
- `DETAIL`: Whether to attach detail latency results of each testcase to the json output. Default value is `True`.

Users could open `<workspace-path>/configs/ruletest_config.yaml` and edit the content. The config will take effect after the config file is saved and closed.

After creating the workspace and completing configuration, users could initialize workspace in `builder_config` module before building the rule_tester:

```python
from nn_meter.builder.utils import builder_config

builder_config.init(
    platform_type="tflite", 
    workspace_path="path/to/workspace/folder"
) # change the text to required platform type and workspace path
```

## Step 2. Create testcases

Following configuration from `<workspace-path>/configs/ruletest_config.yaml`, the test cases can be created by running:

```python
from nn_meter.builder import create_testcases

# generate testcases
origin_testcases = create_testcases()
```

The test case models will be saved in `<workspace-path>/ruletest_testcases/`, and the test case dictionary will be saved in `<workspace-path>/results/origin_testcases.json`.

## Step 3. Run Test Cases on Given Backend

Given required backend, users could run test cases model and get the profiled latency value by running:

```python
from nn_meter.builder.backend import connect_backend
from nn_meter.builder import run_testcases

# initialize backend
backend = connect_backend(backend='tflite_cpu', workspace_path=workspace_path)

# run testcases and collect profiling results
profiled_testcases = run_testcases(backend, origin_testcases)
```
`backend` refers to the framework of the platform and device to execute the model. Currently we provide three instance on two platforms, i.e., CPU backend, GPU backend with TFLite platform, and VPU backend with OpenVINO platform. Refer to [backend guidance](./backend.md) for how to setup the device and get connection to the backend. To use the customized backend, users can follow the [customize backend guidance](./build_customized_backend.md).

The profiled test cases dictionary will be saved in `<workspace-path>/results/profiled_testcases.json`.

## Step 4. Detect Fusion Rule


Finally, users could detect the fusion rule according to the profiled test cases by running:


```python
from nn_meter.builder import detect_fusionrule

# determine fusion rules from profiling results
detected_testcases = detect_fusionrule(profiled_testcases)
```

Two operators $Op1$ and $Op2$ are regarded as being fused as fused as $Op1 +Op2$ fused if the time of operators follows:
$$
T_{Op1} + T_{Op2} - T_{Op1,Op2} > \alpha * min(T_{Op1}, T_{Op2})
$$

After running `detect_fusionrule`, a json file named `<workspace-path>/results/detected_testcases.json` will be created as the detection result. The result shows each test case obeys the fusion rule or not. A instance from the detection result is shown below:

```json
"BF_se_relu": {
    "latency": {
        "block": "20.3537 +- 1.0",
        "se": "20.521 +- 1.0",
        "relu": "2.6194 +- 2.0",
        "ops": "23.1404 +- 2.23606797749979"
    },
    "obey": true
},
...
```
In the results, four `"latency"` value represents the running time of ops `"block"` (which indicates $T_{Op1,Op2}$), two single ops `"se"` ($T_{Op1})$) and `"relu"` ($T_{Op2}$),  and the sum of two ops `"ops"` ($T_{Op1} + T_{Op2}$), respectively. `"obey"` shows whether the test case obeys the fusion rule, with `true` indicates the two testing ops is fused on the backend, while `false` indicates not.

Note that the latency value will be save only when `"DETAIL"` set as `True` in `<workspace-path>/configs/ruletest_config.yaml`.

## Data Structure of TestCases
Each test case consists of several test models to profile, indicating two ops and a block combining the two ops, respectively. In each part, `"model"` points to its directory to the path of this ops' Keras model, `"shapes"` indicates the input shape of the tensor to test, and `"latency"` reports the profiled results after running `run_testcases`. This is a json dump of generated testcases. Note that the `"latency"` attribute appears after running and profiling the test cases.

```json
{
    "BF_dwconv_relu": {
        "dwconv": {
            "model": "./test_models/BF_dwconv_relu_dwconv",
            "shapes": [
                [
                    28,
                    28,
                    16
                ]
            ],
            "latency": "41.781 +- 1.0"
        },
        "relu": {
            "model": "./test_models/BF_dwconv_relu_relu",
            "shapes": [
                [
                    28,
                    28,
                    16
                ]
            ],
            "latency": "2.36618 +- 0.0"
        },
        "block": {
            "model": "./test_models/BF_dwconv_relu_block",
            "shapes": [
                [
                    28,
                    28,
                    16
                ]
            ],
            "latency": "41.4198 +- 1.0"
        }
    },
    ...
}
```
In this instance, `BF_dwconv_relu` is the name of a rule. Here, there are three models called `dwconv`, `relu` and `block`. For each model, the `model` field is the path to where the model is saved. `shapes` is its inputs. For example, here `[[28, 28, 16]]` means this model has only one input, and the shape is `(28, 28, 16)`.

## End-to-end Demo
Here is an end-to-end demo for the progress of the rule tester:
```python
from nn_meter.builder.utils import builder_config
from nn_meter.builder.backends import connect_backend
from nn_meter.builder import create_testcases, run_testcases, detect_fusionrule

# create workspace folder
builder_config.init(
    platform_type="tflite", 
    workspace_path="path/to/workspace/folder"
)

# initialize backend
backend = connect_backend(backend='tflite_cpu', workspace_path=workspace_path)

# generate testcases
origin_testcases = create_testcases()

# run testcases and collect profiling results
profiled_testcases = run_testcases(backend, origin_testcases)

# determine fusion rules from profiling results
detected_testcases = detect_fusionrule(profiled_testcases)
```

After each step, the output will be dumped to `<workspace-path>/results/`. Both the testcases instance and path string to the dumped testcases file are acceptable for the next step.

Also note that it's optional to use a backend. What `run_testcases` do is just collecting latency results of each testcases, so you can use your own tools to measure the latency. Refer to implementation of `run_testcases` for how to fill back the latency.
