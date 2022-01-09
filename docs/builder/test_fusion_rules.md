# Build Fusion Rule Tester

A fusion rule tester creates a series of models (what we call "test case" in nn-Meter). It generates test case models of pairs of operators, profiles the models' latency, and finally, detects the fusion rules for every pair of operators. To build a fusion rule tester, there are four steps to implement the rule detection.

## Step 1. Prepare Backends and Create Workspace

The first step to run fusion rule tester is to prepare backends and create workspace. Users could follow guidance [here](./prepare_backend.md) for this step.

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
    backend_type="tflite", 
    workspace_path="path/to/workspace/folder"
) # change the text to required platform type and workspace path
```
`backend_type` refers to the framework type of the platform. Currently we provide two types of backend, i.e., TFLite platform, and OpenVINO platform. Refer to [backend guidance](./prepare_backend.md) for how to setup the device and get connection to the backend. To use the customized backend, users can follow the [customize backend guidance](./prepare_backend.md#build_customized_backend).

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
backend = connect_backend(backend='tflite_cpu')

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
builder_config.init("tflite", "path/to/workspace/folder") # initialize builder config with workspace
from nn_meter.builder.backends import connect_backend
from nn_meter.builder.backend_meta.fusion_rule_tester import generate_testcases, detect_fusion_rule

# generate testcases
origin_testcases = generate_testcases()

# connect to backend
backend = connect_backend(backend='tflite_cpu')

# run testcases and collect profiling results
profiled_results = profile_models(backend, origin_testcases, mode='ruletest')

# determine fusion rules from profiling results
detected_results = detect_fusion_rule(profiled_results)
```

Three are three main steps, including 1. generate testcase, 2. profile models, and 3. detect fusion rule. For each step, the output will be dumped to `<workspace-path>/fusion_rule_test/results/`. Both the testcases instance and path string to the dumped testcases file are acceptable for the next step.

Note: it's optional to use a backend. What `profile_models` do is collecting latency results of each testcases, so you can use your own tools to measure the latency. Refer to implementation of `profile_models` for how to fill back the latency.

# <span id="test-cases"> Test Cases </span>

Testcases are a series of models created by nn-Meter. These models will be profiled to get latency. By analyzing the latency results, we are able to detect the fusion rules on the device. Finally, the detected fusion rules will be used to direct the process of kernel detection.

In this section, we will explain how our test case classes are implemented and how to customized your own test cases.

## Test Cases Design

Our test case design is driven by two features of a CNN model which impact the fusion rules, i.e., operator type and operator connection 

### Basic Test Cases

Currently, we provide these operators with corresponding name:

- `conv`: layer implemented by `tf.keras.layers.Conv2D`. Input tensor: 3d; Output tensor: 3d.
- `dwconv`: layer implemented by `tf.keras.layers.DepthwiseConv2D`. Input tensor: 3d; Output tensor: 3d.
- `convtrans`: layer implemented by `tf.nn.conv2d_transpose`. Input tensor: 3d; Output tensor: 3d.
- `bn`: layer implemented by `tf.keras.layers.GlobalAveragePooling2D`. Input tensor: 3d; Output tensor: 3d.
- `maxpool`: layer implemented by `tf.keras.layers.MaxPool2D`. Input tensor: 3d; Output tensor: 3d.
- `avgpool`: layer implemented by `tf.keras.layers.AveragePooling2D`. Input tensor: 3d; Output tensor: 3d.
- `global_avgpool`: layer implemented by `tf.keras.layers.GlobalAveragePooling2D`. Input tensor: 3d; Output tensor: 1d.
- `se`: layer implemented refering to [official version](https://github.com/tensorflow/models/blob/89dd9a4e2548e8a5214bd4e564428d01c206a7db/research/slim/nets/mobilenet/conv_blocks.py#L408). Input tensor: 3d; Output tensor: 3d.
- `fc`: layer implemented by `tf.keras.layers.Dense`. Input tensor: 1d; Output tensor: 1d.
- `relu`: layer implemented by `tf.keras.layers.ReLU`. Input tensor: 3d or 1d; Output tensor: 3d or 1d.
- `relu6`: layer implemented by `tf.nn.relu6`. Input tensor: 3d or 1d; Output tensor: 3d or 1d.
- `sigmoid`: layer implemented by `tf.nn.sigmoid`. Input tensor: 3d or 1d; Output tensor: 3d or 1d.
- `hswish`: layer implemented by `tf.nn.relu6`. Input tensor: 3d or 1d; Output tensor: 3d or 1d.
- `reshape`: layer implemented by `tf.reshape`. Input tensor: 3d tensor with shape [H, W, C], or 1d tensor with shape [CIN]; Output tensor: 3d tensor with shape [C, H, W], or 3d tensor with shape [1, 2, CIN / 2]. `CIN` is required to be odd.
- `add`: layer implemented by `tf.keras.layers.Add`. Input tensor: list of two 3d tensor with shape [[H, W, C], [H, W, C]], or 1d tensor with shape [CIN]; Output tensor: one 3d tensor with shape [H, W, C], or one 1d tensor with shape [CIN]. The input tensor will be duplicated as input tensor list.
- `concat`: layer implemented by `tf.keras.layers.Concatenate`. Input tensor: list of two 3d tensor with shape [[H, W, C], [H, W, C]], or 1d tensor with shape [CIN]; Output tensor: one 3d tensor with shape [H, W,  2 * C], or 1d tensor with shape [CIN * 2]. The input tensor will be duplicated as input as input tensor list.
- `flatten`: layer implemented by `tf.keras.layers.Flatten`. Input tensor: 3d; Output tensor: 1d.
- `split`: layer implemented by `tf.split`. Input tensor: 3d; Output tensor: list of two 3d tensor with shape [[H, W, C / 2], [H, W, C / 2]]. `CIN` is required to be odd. 

TODO: In each test cases, we have two ... three models with ... 

Above ops can be used for fusion rule testing of single inbound and outbound operator connections. Users could edit `'BASIC_TESTCASES'` in `<workspace-path>/configs/ruletest_config.yaml` to determine the interested combination. A demo of `'BASIC_TESTCASES'` is:

```json
BASIC_TESTCASES:
  - conv_avgpool
  - conv_relu
```

which indicates that in the progress of fusion rule detection, two test cases will be generated, including the connection of `conv` op and `avgpool` op, as well as the connection of `conv` op and `avgpool` op. To add new test case, users could use the layer name, connect the op names by `"_"` and add the string to `'BASIC_TESTCASES'`.

Note: if the input to this layer is 1 dimension tensor, users should add it into `'LAYERS_1D'` in `<workspace-path>/configs/ruletest_config.yaml`.

### Other Test Cases

Besides of operator type, operator connection also impacts fusion rules. nn-Meter composed three basic connection types, including 1. single inbound and out bound, 2. multiple outbounds, and 3. multiple inbounds.

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

On the other side, to test multiple inbounds, 
```python
input_layer = keras.Input(shape=self.input_shape)

branch_1 = keras.layers.DepthwiseConv2D(self.kernel_size, padding=self.padding)(input_layer)
branch_2 = keras.layers.DepthwiseConv2D(self.kernel_size, padding=self.padding)(input_layer)
output_1 = keras.layers.Add()([branch_1, branch_2])
branch_3 = keras.layers.DepthwiseConv2D(self.kernel_size, padding=self.padding)(input_layer)
output_1 = keras.layers.Add()([branch_3, output_1])

output_2 = keras.layers.ReLU()(branch_3)

return keras.Model(input_layer, [output_1, output_2])
```

## Data Structure of Test Cases

Each test case consists of several test models to profile. Generally, for basic test cases, test models indicates two ops and a block combining the two ops. In each test models, `"model"` points to its directory to the path of this ops' `Keras` model, `"shapes"` indicates the input shape of the tensor to test, and `"latency"` reports the profiled results after running `run_testcases`. This is a json dump of generated test cases. Note that the `"latency"` attribute appears after running and profiling the test cases.

```json
{
    "dwconv_relu": {
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
In this instance, `dwconv_relu` is the name of a test case. There are three models called `dwconv`, `relu` and `block`. For each model, the `"model"` indicates the path to where the model is saved. `"shapes"` indicates the list of the its input tensor shape (`[H, W, C]`). For example, here `[[28, 28, 16]]` means this model has only one input, and the shape is `(28, 28, 16)`.

# Apply Fusion Rules for Kernel Detection

TODO


# <span id="build-customized-test-cases"> Build Customized Test Cases </span>

## Build Basic Test cases

`nn_meter.builder.backend_meta.fusion_rule_tester.generate_testcases.TestCasesGenerator` is the base of all rules. We define default behaviors in this base class. There are following methods:

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

- `_register`: Only rules subclassing `RuleTestBase` will be registered into `rules`. And only these rules will be generated and profiled. You can access that all rules by `nn_meter.builder.backend_meta.fusion_rule_tester.rules.rules`.

### declare different basic fusion test rules
If you want to add new operators, just subclass `BasicFusion` and add new op to `layers`:

```python
layers = [
    'reshape',
    'dwconv',
    'relu',
    'add',
    'conv',
    'concat',
    'convtrans',
    'dense',
    'pooling',
    'hswish',
]
```

### add new operator to basic fusion
Add New Operators into BasicFusion


## build other test case

Customized rules are more complicated. What you need to do is subclassing `ReadyTensor`.

If you are satisfied with the default behavior of each function described in [implementation](#implementation), then you only need to define following class members by following this example:

```python

class ReadyTensor(TestCasesGenerator):
    name = 'RT'
    cases = {
        'case1': ['dwconv_add', 'dwconv', 'dwconv', 'add', 'relu'],
        'case2': ['dwconv_add_add', 'dwconv', 'dwconv', 'relu'],
    }
    true_case = 'case1'
    deps = {
        'Multiple_Outbounds': True,
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

- `name`: the name of the rule

- `_model_block`: The structure of the tested block.

- `cases`: The potential splitting possibility of `_model_block`.

- `deps`: The truth of this rule will depend on truth of other rules.

BasicFusion is more complicated. From design, each rule will generate testcases for its rule and analyze to decide whether the rule obeys or not. But for BasicFusion, it has a lot of variance, e.g., `BF_conv_relu` to test the fusion rule of convolution and relu.

We implement this by subclassing. We automatically generate a lot of subclasses of `BasicFusion` on the fly by some python magic (metaclass). Each subclass refers to the fusion rule of one pair of operators.

## Use Customized Rules when Splitting

Currently we haven't provided api to split models using customized rules. We leave that to future work.

It's not suggested, but you can implement that by directly modifying the code at `nn_meter.kernel_detector.rulelib.rule_splitter`.



