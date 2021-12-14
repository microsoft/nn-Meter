# Testcase

Testcases are a series of models. These models will be profiled to get latency. By analyzing the latency results, we are able to detect the fusion rules on the device. For example, `BasicFusion` will detect fusion possibility of each pair of two operators. Finally, the detected fusion rules will be used to direct the process of kernel detection.

In this section, we will explain how our testcase classes are implemented and how you can provide your own testcases.

## Implementation

`nn_meter.builder.rule_tester.rules.RuleTestBase` is the base of all rules. We define default behaviours in this base class. There are following methods:

- `generate_testcase`: Generate all testcase models for this rule. At the **most time you won't** need to modify this.

- `save_testcase`: Save the testcases. The `_model_block` of rule `Rule1` will be saved as name `Rule1_block`. At the **most time you won't** need to modify this.

- `load_latency`: Load the latency from a json results. At **the most time you won't** need to modify this.

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

- `_register`: Only rules subclassing `RuleTestBase` will be registered into `rules`. And only these rules will be generated and profiled. You can access that all rules by `nn_meter.builder.rule_tester.rules.rules`.

### BasicFusion

BasicFusion is more complicated. From design, each rule will generate testcases for its rule and analyze to decide whether the rule obeys or not. But for BasicFusion, it has a lot of variance, e.g., `BF_conv_relu` to test the fusion rule of convolution and relu.

We implement this by subclassing. We automatically generate a lot of subclasses of `BasicFusion` on the fly by some python magic (metaclass). Each subclass refers to the fusion rule of one pair of operators.

## Customized Rules

### Add New Operators into BasicFusion

Currently we only test fusion rules among these layers (or operators):

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
]
```

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
    'hswish', # add a new op
]
```

If the input to this layer must be 1 dimension tensor, you need also add it into `d1_required_layers`.

If you only need to add one combination, you can add that into `additional_combinations`:
```python
additional_combinations = [
    ('conv', 'hswish'),
    ('conv', 'se'),
    ('se', 'relu'),
]
```

### Other rules

Customized rules are more complicated. What you need to do is subclassing `ReadyTensor`.

If you are satisfied with the default behaviour of each function described in [implementation](#implementation), then you only need to define following class members by following this example:

```python
class ReadyTensor(RuleTestBase):
    name = 'RT'
    cases = {
        'case1': ['dwconv_add', 'dwconv', 'dwconv', 'add', 'relu'],
        'case2': ['dwconv_add_add', 'dwconv', 'dwconv', 'relu'],
    }
    true_case = 'case1'
    deps = {
        'MON': True,
        'BF_dwconv_relu': True,
    }
    def _model_block(self):
        input_layer = keras.Input(shape=self.input_shape)

        branch_1 = keras.layers.DepthwiseConv2D(self.kernel_size, padding='same')(input_layer)
        branch_2 = keras.layers.DepthwiseConv2D(self.kernel_size, padding='same')(input_layer)
        output_1 = keras.layers.Add()([branch_1, branch_2])
        branch_3 = keras.layers.DepthwiseConv2D(self.kernel_size, padding='same')(input_layer)
        output_1 = keras.layers.Add()([branch_3, output_1])

        output_2 = keras.layers.ReLU()(branch_3)

        return keras.Model(input_layer, [output_1, output_2]), [self.input_shape]
```

- `name`: the name of the rule

- `_model_block`: The structure of the tested block.

- `cases`: The potential splitting possibility of `_model_block`.

- `deps`: The truth of this rule will depend on truth of other rules.


## Use Customized Rules when Splitting

Currently we haven't provided api to split models using customized rules. We leave that to future work.

It's not suggested, but you can implement that by directly modifying the code at `nn_meter.kernel_detector.rulelib.rule_splitter`.
