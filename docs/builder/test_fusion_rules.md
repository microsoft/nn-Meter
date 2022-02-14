# Build Fusion Rule Tester

A fusion rule tester creates a series of models (what we call "test case" in nn-Meter). It generates test case models of pairs of operators, profiles the models' latency, and finally, detects the fusion rules for every pair of operators. To build a fusion rule tester, there are four steps to implement the rule detection.

## Step 1. Prepare Backends and Create Workspace

The first step to run fusion rule tester is to prepare backends and create workspace. Users could follow guidance [Prepare Backends](./prepare_backend.md) and [Create Workspace](./overview.md#create-workspace) for this step.

After creating the workspace, a yaml file named `ruletest_config.yaml` will be placed in `<workspace-path>/configs/`. The fusion rule test configs includes:

- `HW`: Default input shape of all test cases except those requiring 1d tensor input. Default value is `28`.
- `CIN`: Default input channel of all test cases. Default value is `16`.
- `SHAPE_1D`: Default input shape of all testcases that need 1d tensor input. E.g., fully connected layer. Default value is `428`.
- `COUT`: Default output channel (filter size). Default value is `256`.
- `KERNEL_SIZE`: Default kernel size. Default value is `3`.
- `PADDING`: Default padding type. Default value is `"same"`.
- `STRIDES`: Default strides size. Default value is `1`.
- `POOL_STRIDES`: Default strides size for pooling operator. Default value is `2`.
- `EPS_ALPHA`: The empirical coefficient as a threshold in formula of [step 4](#step-4-detect-fusion-rule) to decide whether two ops can be fused for test cases of BasicFusion. Default value is `0.5`.
- `DETAIL`: Whether to attach detail information to the json output, such as the shape information in profiled results, and the latency results of each test case in detected fusion rules. Default value is `FALSE`.
- `BASIC_TESTCASES`: the test cases list to test. Generally, there are three types of test cases. Basic test cases detect the fusion rule of single inbound and outbound operators pairs. 
- `OTHER_TESTCASES`: in this list, `'MON'` detects the fusion rules about multiple outbounds connection. Besides, users can add name of customized test cases after test cases registration. For more details refer to the introduction of [Test Cases](#test-cases) and [Build Customized Test Cases](#build-customized-test-cases).
- `LAYERS_1D`: the list of layer name for 1d tensor input. If the input to this layer must be 1 dimension tensor, users need add it here.

Note: nn-Meter doesn't support different `"COUT"` parameters for conv layer. If there are two successive convolutional layers in the test case, the output channel of the two layer will be the same as `"COUT"` parameter.

Users could open `<workspace-path>/configs/ruletest_config.yaml` and edit the content. After completing configuration, users could initialize workspace in `builder_config` module before building the fusion rule tester:

```python
from nn_meter.builder.utils import builder_config

# initialize builder config with workspace
builder_config.init(
    workspace_path="path/to/workspace/folder"
) # change the text to required platform type and workspace path
```

Note: after running ``builder_config.init``, the config are loaded already. If users want to update config, after the updated config file is saved and closed, the config will take effect after reload config space by running ``builder_config.init`` again.

## Step 2. Create testcases

Following configuration from `<workspace-path>/configs/ruletest_config.yaml`, the test cases can be created by running:

```python
from nn_meter.builder.backend_meta.fusion_rule_tester import generate_testcases

# generate testcases
origin_testcases = generate_testcases()
```

The test case models will be saved in `<workspace-path>/fusion_rule_test/models/`, and the information of test cases will be saved in `<workspace-path>/fusion_rule_test/results/origin_testcases.json`.

## Step 3. Run Test Cases on Given Backend

Given required backend, users could run test cases model and get the profiled latency value by running:

```python
# connect to backend
from nn_meter.builder.backends import connect_backend
backend = connect_backend(backend_name='tflite_cpu')

# run testcases and collect profiling results
from nn_meter.builder import profile_models
profiled_results = profile_models(backend, origin_testcases, mode='ruletest')
```
`backend` refers to the name of concrete device to execute the model. Currently we provide three devoce instance, i.e., CPU backend, GPU backend with TFLite platform, and VPU backend with OpenVINO platform. Refer to [backend guidance](./prepare_backend.md) for how to setup the device and get connection to the backend. To use the customized backend, users can follow the [customize backend guidance](./prepare_backend.md#build_customized_backend).

The profiled test cases dictionary will be saved in `<workspace-path>/fusion_rule_test/results/profiled_results.json`.

## <span id="step-4-detect-fusion-rule"> Step 4. Detect Fusion Rule </span>

Finally, users could detect the fusion rule according to the profiled test cases by running:

```python
from nn_meter.builder.backend_meta.fusion_rule_tester import detect_fusion_rule

# determine fusion rules from profiling results
detected_results = detect_fusion_rule(profiled_results)
```

Two operators $Op1$ and $Op2$ are regarded as being fused as fused as $Op1 +Op2$ fused if the time of operators follows:
$$
T_{Op1} + T_{Op2} - T_{Op1,Op2} > \alpha * min(T_{Op1}, T_{Op2})
$$

After running `detect_fusion_rule`, a json file named `<workspace-path>/fusion_rule_test/results/detected_results.json` will be created to save the detection result. The result shows each test case obeys the fusion rule or not. A instance from the detection result is shown below:

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

Note: the latency value will be recorded only when `'DETAIL'` set as `True` in `<workspace-path>/configs/ruletest_config.yaml`.

## End-to-end Demo

Here is an end-to-end demo for the progress for the fusion rule testing:

```python
from nn_meter.builder import profile_models
from nn_meter.builder.utils import builder_config
builder_config.init("path/to/workspace/folder") # initialize builder config with workspace
from nn_meter.builder.backends import connect_backend
from nn_meter.builder.backend_meta.fusion_rule_tester import generate_testcases, detect_fusion_rule

# generate testcases
origin_testcases = generate_testcases()

# connect to backend
backend = connect_backend(backend_name='tflite_cpu')

# run testcases and collect profiling results
profiled_results = profile_models(backend, origin_testcases, mode='ruletest')

# determine fusion rules from profiling results
detected_results = detect_fusion_rule(profiled_results)
```

Three are three main steps, including 1) generate testcase, 2) profile models, and 3) detect fusion rule. For each step, the output will be dumped to `<workspace-path>/fusion_rule_test/results/`. Both the testcases instance and path string to the dumped testcases file are acceptable for the next step.

Note: it's optional to use a backend. What `profile_models` do is collecting latency results of each testcases, so you can use your own tools to measure the latency. Refer to implementation of `profile_models` for how to fill back the latency.

# <span id="test-cases"> Test Cases </span>

Testcases are a series of models created by nn-Meter. These models will be profiled to get latency. By analyzing the latency results, we are able to detect the fusion rules on the device. Finally, the detected fusion rules will be used to direct the process of kernel detection.

In this section, we will explain how our test case classes are implemented and how to customized your own test cases.

## Test Cases Design

Our test case design is driven by two features of a CNN model which impact the fusion rules, i.e., operator type and operator connection 

### <span id="basic-test-cases"> Basic Test Cases </span>

Currently, we provide these operators with corresponding name:

- `conv`: conv2d layer implemented by `tf.keras.layers.Conv2D`. Input tensor: 3d; Output tensor: 3d.
- `dwconv`: dwconv2d layer implemented by `tf.keras.layers.DepthwiseConv2D`. Input tensor: 3d; Output tensor: 3d.
- `convtrans`: conv2d transpose layer implemented by `tf.nn.conv2d_transpose`. Input tensor: 3d; Output tensor: 3d.
- `bn`: batch normalization layer implemented by `tf.keras.layers.GlobalAveragePooling2D`. Input tensor: 3d; Output tensor: 3d.
- `maxpool`: max pooling layer implemented by `tf.keras.layers.MaxPool2D`. Input tensor: 3d; Output tensor: 3d.
- `avgpool`: average pooling layer implemented by `tf.keras.layers.AveragePooling2D`. Input tensor: 3d; Output tensor: 3d.
- `globalavgpool`: global average pooling layer implemented by `tf.keras.layers.GlobalAveragePooling2D`. Input tensor: 3d; Output tensor: 1d.
- `se`: squeeze excite block implemented refering to [official version](https://github.com/tensorflow/models/blob/89dd9a4e2548e8a5214bd4e564428d01c206a7db/research/slim/nets/mobilenet/conv_blocks.py#L408). Input tensor: 3d; Output tensor: 3d.
- `fc`: fully connection layer implemented by `tf.keras.layers.Dense`. Input tensor: 1d; Output tensor: 1d.
- `relu`: relu activation layer implemented by `tf.keras.layers.ReLU`. Input tensor: 3d or 1d; Output tensor: 3d or 1d.
- `relu6`: relu5 activation layer implemented by `tf.nn.relu6`. Input tensor: 3d or 1d; Output tensor: 3d or 1d.
- `sigmoid`: sigmoid activation layer implemented by `tf.nn.sigmoid`. Input tensor: 3d or 1d; Output tensor: 3d or 1d.
- `hswish`: hswish activation layer implemented by `tf.nn.relu6`. Input tensor: 3d or 1d; Output tensor: 3d or 1d.
- `reshape`: reshape layer implemented by `tf.reshape`. Input tensor: 3d tensor with shape [H, W, C], or 1d tensor with shape [CIN]; Output tensor: 3d tensor with shape [C, H, W], or 3d tensor with shape [1, 2, CIN / 2]. `CIN` is required to be odd.
- `add`: add layer implemented by `tf.keras.layers.Add`. Input tensor: list of two 3d tensor with shape [[H, W, C], [H, W, C]], or 1d tensor with shape [CIN]; Output tensor: one 3d tensor with shape [H, W, C], or one 1d tensor with shape [CIN]. The input tensor will be duplicated as input tensor list.
- `concat`: concatenation layer implemented by `tf.keras.layers.Concatenate`. Input tensor: list of two 3d tensor with shape [[H, W, C], [H, W, C]], or 1d tensor with shape [CIN]; Output tensor: one 3d tensor with shape [H, W,  2 * C], or 1d tensor with shape [CIN * 2]. The input tensor will be duplicated as input as input tensor list.
- `flatten`: flatten layer implemented by `tf.keras.layers.Flatten`. Input tensor: 3d; Output tensor: 1d.
- `split`: split layer implemented by `tf.split`. Input tensor: 3d; Output tensor: list of two 3d tensor with shape [[H, W, C / 2], [H, W, C / 2]]. `CIN` is required to be odd. 

Above ops can be used for fusion rule testing of single inbound and outbound operator connections, which we also call it by "basic test case". In each basic test cases, there are three models generated, including two models containing single op respectively, and a model containing the block of two ops. The test case will test if the two ops will be fused as a block in inference. Users could edit `'BASIC_TESTCASES'` in `<workspace-path>/configs/ruletest_config.yaml` to determine the interested combination. A demo of `'BASIC_TESTCASES'` is:

```json
BASIC_TESTCASES:
  - conv_avgpool
  - conv_relu
```

which indicates that in the progress of fusion rule detection, two test cases will be generated, including the test case of `conv` op and `avgpool` op, as well as the test case of `conv` op and `avgpool` op. To add new test case, users could use the layer name, connect the op names by `"_"` and add the string to `'BASIC_TESTCASES'`.

Note: if the input to this layer is 1 dimension tensor, users should add it into `'LAYERS_1D'` in `<workspace-path>/configs/ruletest_config.yaml`.

### Other Test Cases

Besides of operator type, operator connection also impacts fusion rules. nn-Meter composed three basic connection types, including 1) single inbound and out bound, 2) multiple outbounds, and 3) multiple inbounds. Our study have shown that there will not be any fusion for multiple inbound type, so that we didn't provide any test case for it.

To test multiple outbounds, nn-Meter formed a test case with two branches, named `'MON'`(multiple out nodes). The implementation of the test case block is shown below:

```python
input_layer = keras.Input(shape=input_shape)
x = keras.layers.DepthwiseConv2D(kernel_size, padding=padding)(input_layer)
branch_1 = keras.layers.ReLU(negative_slope=0)(x)
branch_1 = keras.layers.ReLU(negative_slope=0)(branch_1)
branch_2 = keras.layers.ReLU(negative_slope=2)(x)
branch_2 = keras.layers.DepthwiseConv2D(kernel_size, padding=padding)(branch_2)
return keras.models.Model(input_layer, [branch_1, branch_2])
```

If there is a rule exists that `"dwconv_relu"` will be fused as a kernel, there are three cases for multiple outbounds kernel fusion, that is:
```python
cases = {
    'case1': ['relu_relu', 'relu_dwconv', 'dwconv'],
    'case2': ['dwconv_relu_relu', 'relu_dwconv'],
    'case3': ['dwconv_relu', 'dwconv', 'relu_relu']
}
```
we need to test which fusion rule will the test case obey. The detection result of multiple outbounds test case will be a string from `['case1', 'case2', 'case3']`.

## Data Structure of Test Cases

Each test case consists of several test models to profile. Generally, for basic test cases, test models indicates two ops and a block combining the two ops. In each test models, `"model"` points to its directory to the path of this ops' `Keras` model, `"shapes"` indicates the input shape of the tensor to test, and `"latency"` reports the profiled results after running `run_testcases`. This is a json dump of generated test cases. Note that the `"latency"` attribute appears after running and profiling the test cases.

```json
{
    "dwconv_relu": {
        "dwconv": {
            "model": "./fusion_rule_test/models/BF_dwconv_relu_dwconv",
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
            "model": "./fusion_rule_test/models/BF_dwconv_relu_relu",
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
            "model": "./fusion_rule_test/models/BF_dwconv_relu_block",
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

In this instance, `dwconv_relu` is the name of a test case. There are three models called `dwconv`, `relu` and `block`. For each model, the `"model"` indicates the path to where the model is saved. In the name of model, `"BF"` indicates the test case belong to a basic fusion test case, `"dwconv_relu"` indicates the name of the test case, and the last clause (`"dwconv"`, `"relu"`, or `"block"`) indicates the model name in that test case. `"shapes"` indicates the list of the its input tensor shape (`[H, W, C]`). For example, here `[[28, 28, 16]]` means this model has only one input, and the shape is `(28, 28, 16)`.

# Apply Fusion Rules for Kernel Detection

The output json file `<workspace-path>/fusion_rule_test/results/detected_fusion_rule.json` shows all fusion rule detected from test cases. Users could directly apply the json file for kernel detection.

The fusion rules json file will be a part of the customized predictor. Users could refer to [Customize Predictor](../predictor/customize_predictor.md) to prepare other parts of predictor and register predictor.

# <span id="build-customized-test-cases"> Build Customized Test Cases </span>

## Build Basic Test cases

Currently, nn-Meter support the following ops:

- `'conv'`

- `'dwconv'`

- `'convtrans'`

- `'bn'`

- `'maxpool'`

- `'avgpool'`

- `'globalavgpool'`

- `'se'`

- `'fc'`

- `'relu'`

- `'relu6'`

- `'sigmoid'`

- `'hswish'`

- `'reshape'`

- `'add'`

- `'concat'`

- `'flatten'`

- `'split'`

Refer to [basic test cases](#basic-test-cases) for more details of supporting ops. To apply existing ops, users could directly declare the op name and ops connection in `'BASIC_TESTCASES'` from `<workspace-path>/configs/ruletest_config.yaml` to generate their own test cases.

If users want to add new operators into basic test cases, here are several steps to prepare and register new operators:

### Step 1: Prepare the Customized Operator Class

nn-Meter provide API for users to customize their own operator. In nn-Meter, each operator is implemented by inheriting a base class named `nn_meter.builder.nn_generator.tf_networks.BaseOperator`. The class has two input parameters, including `input_shape` and `config`. `input_shape` is a list showing the dimension of the input tensor (the batch dimension should not be included), and `config` can be used to feed configuration params for the operator. There are following methods in this base class:

- `get_model`: Return the model function of the operator. Users need to modify this **all the time**.

- `get_output_shape`: Return a list representing the output shape of the operator. Users need to modify this **at the most time**. If the output has same shape with input, users don't need to override the method.

- `get_is_two_inputs`: Whether the operator has two input tensor. If the operator has only one input tensor, the returned value should be set as `False`. For **some time you will** need to modify this. If the value is `False`, users don't need to override the method.

- `test_operator`: A test script to verify the operator. At the **most time you won't** need to modify this.

We provide the implementation of three builtin operators for example:

The first example is Conv2d operator. The operator simply applying APIs from `tensorflow.keras.layers` to build the model function. Also, users could build a class inheriting `tensorflow.keras.layers.Layer` for customized usage:

``` python
class Conv(BaseOperator):
    def get_model(self):
        cout = self.input_shape[2] if "COUT" not in self.config else self.config["COUT"]
        return keras.layers.Conv2D(
            cout,
            kernel_size=self.config["KERNEL_SIZE"],
            strides=self.config["STRIDES"],
            padding="same"
        )

    def get_output_shape(self):
        cout = self.input_shape[2] if "COUT" not in self.config else self.config["COUT"]
        output_h = (self.input_shape[0] - 1) // self.config["STRIDES"] + 1
        output_w = (self.input_shape[1] - 1) // self.config["STRIDES"] + 1
        return [output_h, output_w, cout]
```

The second example is Sigmoid operator. The opertor build a model function by applying `tensorflow.nn`:

``` python
class Sigmoid(BaseOperator):
    def get_model(self):
        def func(inputs):
            return tf.nn.sigmoid(inputs)
        return func
```

The third example is an operator class with two input tensor. In this case, users should notice that the `output_shape` must cover all probably cases for `input_shape`:

``` python
class Add(BaseOperator):
    def get_model(self):
        return keras.layers.Add()

    def get_output_shape(self):
        if len(self.input_shape) == 2 and type(self.input_shape[0]) == list:
            output_shape = self.input_shape[0]
        else:
            output_shape = self.input_shape
        return output_shape

    def get_is_two_inputs(self):
        return True
```

Note: all configuration value are feed into the operators by the param `config`, which gets data from `<workspace-path>/configs/ruletest_config.yaml`. Users should follow the same notation of configs to transfer parameters. If there is any new parameters needed in `config`, users should also add the parameter and set its value in `<workspace-path>/configs/ruletest_config.yaml`.

### Step 2: Create a Package for the Customized Operator

nn-Meter requires users to gather all code of operator in a package with a fixed location. A folder will be treated as a package with a `__init__.py` file added. Here is a demo of folder structure:

``` text
./customized_operator/
├── __init__.py
└── operator_script.py
```

The interface of customized operator class are stored in `./customized_operator/operator_script.py`. In this demo, the content of `operator_script.py` includes:

``` python
from nn_meter.builder.nn_generator.tf_networks import BaseOperator
from tensorflow import keras

def Op1(BaseOperator):
    def get_model(self):
        return ...
```

Note: The folder could contain more than one operators, but the registration should be done one by one.

### Step 3: Prepare Meta File

Create a yaml file with following keys as meta file:

- `builtin_name`: builtin name used in nn-Meter configuration file to call the customized operator, such as `"op1"`. Note that there should not have any "\_" in the `buildinName`, as any "\_" will be regarded as the connection of different operators in test cases generation.

- `package_location`: the absolute path of the package folder.

- `class_module`: the module of the operator class, in this example is `operator_script`, representing `operator_script.py`.

- `class_name`: the operator class name, in this example is `Op1`.

Following is an example of the yaml file:

```yaml
builtin_name: op1
package_location: /home/USERNAME/working/customized_operator
class_module: operator_script
class_name: Op1
```

### Step 4: Register Customized Operator into nn-Meter

Run the following command to register customized operator into nn-Meter:

``` bash
nn-meter register --operator path/to/meta/file
```
If the registration success, nn-Meter will show:
``` text
(nn-Meter) Successfully register operator: op1
```

When registering, nn-Meter will test whether the module can be imported first. If the registration success is not successful, please check the package according to the error information.

After backend registration, users can view all operators by running:
``` bash
nn-meter --list-operators
```
```text
(nn-Meter) Supported operators: ('*' indicates customized operators)
(nn-Meter) [Operator] conv
(nn-Meter) [Operator] dwconv
(nn-Meter) [Operator] convtrans
(nn-Meter) [Operator] bn
(nn-Meter) [Operator] globalavgpool
(nn-Meter) [Operator] maxpool
(nn-Meter) [Operator] avgpool
(nn-Meter) [Operator] se
(nn-Meter) [Operator] fc
(nn-Meter) [Operator] relu
(nn-Meter) [Operator] relu6
(nn-Meter) [Operator] sigmoid
(nn-Meter) [Operator] hswish
(nn-Meter) [Operator] reshape
(nn-Meter) [Operator] add
(nn-Meter) [Operator] concat
(nn-Meter) [Operator] flatten
(nn-Meter) [Operator] split
(nn-Meter) [Operator] * op1
```

Note: the package of customized operator must be retained in a fixed path as registered one. Otherwise may cause error when calling the registered module.

### Use the Customized Operator in Experiment

After registration, users could apply the customized operator to generate test case:

``` yaml
# .yaml file in `<workspace-path>/configs/ruletest_config.yaml`
...
BASIC_TESTCASES:
  - op1_relu
LAYERS_1D:
  - fc
  - op1
```

Note: if the input to customized operator should be 1 dimension tensor, users need add the builtin name `LAYERS_1D`.

### Manage the Registered Operator

Users could unregister the operator by calling its name in command:

``` bash
nn-meter unregister --operator op1
```
``` text
(nn-Meter) Successfully unregister op1.
```

After unregister the operator, "op1" will be removed from the backend list.

## Build Other Test Case

### Step 1: Prepare the Customized Test Case Class
Customized other test case are more complicated. Here we describe the implementation of `BaseTestCase` first. We define the base of all test case generator in `nn_meter.builder.backend_meta.fusion_rule_tester.BaseTestCase`. There are following methods in this base class:

- `generate_testcase`: Generate all test models for this test case. At the **most time you won't** need to modify this.

- `save_testcase`: Save the test cases models and return the test cases information. The `_model_block` of rule `Rule1` will be saved as name `Rule1_block`. At the **most time you won't** need to modify this.

- `load_latency`: Load the latency from test case information (usually shown as a dictionary class in json format). At **the most time you won't** need to modify this.

- `test`: Decide the truth or case of this rule by analyzing latency results. For **some time you will** need to modify this.

- `load_config`: Load configuration that will be used in the class. At the **most time you won't** need to modify this.

- Methods starting with `_model_`: It is used to define the structure of model in testcases. For example, if you define `_model_conv` in the class, then you can use `conv` in field `cases`. This means `conv` will be generated as a model, profiled and used for latency analysis as a component of the case used in. For example,
  
    ```python
    cases = {
        'case1': ['dwconv_add', 'dwconv', 'dwconv', 'add', 'relu'],
        'case2': ['dwconv_add_add', 'dwconv', 'dwconv', 'relu'],
    }
    ```

    Here latency of `case1` is the sum of latency of `_model_dwconv_add`, `_model_dwconv` * 2, `_model_add`, `_model_relu`.

    **For all the time you will** need to implement `_model_block`.

Besides, there are also several class attributes:

- `name`: the name of the fusion rule

- `cases`: The potential splitting possibility of `_model_block`.

- `deps`: The truth of this rule will depend on truth of other rules.

- `_model_block`: The structure of the tested block.

Here is an example for customized test case:

```python
from nn_meter.builder.backend_meta.fusion_rule_tester import BaseTestCase

class MyTestCase(BaseTestCase):
    name = 'MyTestCase'
    cases = {
        'case1': ['dwconv_add', 'dwconv', 'dwconv', 'add', 'relu'],
        'case2': ['dwconv_add_add', 'dwconv', 'dwconv', 'relu'],
    }
    true_case = 'case1'
    deps = {
        'MON': True,
        'dwconv_relu': True,
    }

    def _model_block(self):
        input_layer = keras.Input(shape=self.input_shape)

        branch_1 = keras.layers.DepthwiseConv2D(self.kernel_size, padding=self.padding)(input_layer)
        branch_2 = keras.layers.DepthwiseConv2D(self.kernel_size, padding=self.padding)(input_layer)
        output_1 = keras.layers.Add()([branch_1, branch_2])
        branch_3 = keras.layers.DepthwiseConv2D(self.kernel_size, padding=self.padding)(input_layer)
        output_1 = keras.layers.Add()([branch_3, output_1])

        output_2 = keras.layers.ReLU()(branch_3)

        return keras.Model(input_layer, [output_1, output_2]), [self.input_shape]

    def _model_dwconv_add(self):
        input_layer = keras.Input(shape=self.input_shape)

        x = keras.layers.DepthwiseConv2D(self.kernel_size, padding=self.padding)(input_layer)
        x = keras.layers.Add()([x, input_layer])

        return keras.models.Model(input_layer, x), [self.input_shape]

    def _model_dwconv_add_add(self):
        input_1 = keras.Input(shape=self.input_shape)
        input_2 = keras.Input(shape=self.input_shape)

        x = keras.layers.DepthwiseConv2D(self.kernel_size, padding=self.padding)(input_1)
        x = keras.layers.Add()([x, input_1])
        x = keras.layers.Add()([x, input_2])

        return keras.models.Model([input_1, input_2], x), [self.input_shape, self.input_shape]
```

### Step 2: Create a Package for the Customized Test Case

nn-Meter requires users to gather all code of test case in a package with a fixed location. A folder will be treated as a package with a `__init__.py` file added. Here is a demo of folder structure:

``` text
./customized_testcase/
├── __init__.py
└── testcase_script.py
```

The interface of customized test case class are stored in `./customized_testcase/testcase_script.py`.

### Step 3: Prepare Meta File

Create a yaml file with following keys as meta file:

- `builtin_name`: builtin name used in nn-Meter configuration file to call the customized test case, such as `"MyTC"`.

- `package_location`: the absolute path of the package folder.

- `class_module`: the module of the test case class, in this example is `testcase_script`, representing `testcase_script.py`.

- `class_name`: the test case class name, in this example is `MyTestCase`.

Following is an example of the yaml file:

```yaml
builtin_name: MyTC
package_location: /home/USERNAME/working/customized_testcase
class_module: testcase_script
class_name: MyTC
```

### Step 4: Register Customized Test Case into nn-Meter

Run the following command to register customized test case into nn-Meter:

``` bash
nn-meter register --testcase path/to/meta/file
```
If the registration success, nn-Meter will show:
``` text
(nn-Meter) Successfully register testcase: MyTC
```

When registering, nn-Meter will test whether the module can be imported first. If the registration success is not successful, please check the package according to the error information.

Note: the package of customized operator must be retained in a fixed path as registered one. Otherwise may cause error when calling the registered module.

### Use the Customized Operator in Experiment

After registration, users could apply the customized operator to generate test case:

``` yaml
# .yaml file in `<workspace-path>/configs/ruletest_config.yaml`
...
OTHER_TESTCASES:
  - MyTC
```

Note: if the truth of this fusion rule depends on truth of other rules, the test of precondition must be done first. That is, the fusion rule in attribute `deps` must be tested before or with the customized test case.

### Manage the Registered Operator

Users could unregister the test case by calling its name in command:

``` bash
nn-meter unregister --testcase MyTC
```
``` text
(nn-Meter) Successfully unregister MyTC.
```

## Use Customized Rules when Splitting

Currently we haven't provided api to split models using customized rules. We leave that to future work.

It's not suggested, but you can implement that by directly modifying the code at `nn_meter.kernel_detector.rule_splitter`.
