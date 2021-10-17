# RuleTest Builder

This part generates and profiles models (what we call "test cases") on a given device to get rules.

## End-to-End Demo

```python
from nn_meter import get_testcases, run_testcases, get_fusionrule
from nn_meter.backends.utils import get_backend

# generate testcases
testcases = get_testcases('~/test_models')
# initialize backend
backend = get_backend('tflite_cpu', {
    'MODEL_DIR': '',
    'REMOTE_MODEL_DIR': '',
    'KERNEL_PATH': '',
    'BENCHMARK_MODEL_PATH': '',
    'DEVICE_SERIAL': '',
})
# run testcases and collect profiling results
profile_results = run_testcases(backend, testcases)
# determine fusion rules from profiling results
result = get_fusionrule(profile_results)
```

`backend` refers to device and framework to execute the model. Currently we support TFLite on CPU, GPU and OpenVINO on VPU. Refer to [backend guide](./backend.md) for how to setup the device and backend.

Note that we provide a consistent API for backend, so you can inherit the `BaseBackend` and implement a backend for your own device that's not included here. 

Also note that it's optional to use a backend. What `run_testcases` do is just collecting latency results of each testcases, so you can use your own tools to measure the latency. Refer to implementation of `run_testcases` for how to fill back the latency.

## Data Structure of TestCases

This is a json dump of generated testcases.

```json
{
    "BF_reshape_reshape": {
        "reshape_1": {
            "model": "/Users/kalineid/test_models/BF_reshape_reshape_reshape_1",
            "shapes": [
                [
                    28,
                    28,
                    16
                ]
            ]
        },
        "reshape_2": {
            "model": "/Users/kalineid/test_models/BF_reshape_reshape_reshape_2",
            "shapes": [
                [
                    16,
                    28,
                    28
                ]
            ]
        },
        "block": {
            "model": "/Users/kalineid/test_models/BF_reshape_reshape_block",
            "shapes": [
                [
                    28,
                    28,
                    16
                ]
            ]
        }
    },
    "BF_reshape_dwconv": {
        "reshape": {
            "model": "/Users/kalineid/test_models/BF_reshape_dwconv_reshape",
            "shapes": [
                [
                    28,
                    28,
                    16
                ]
            ]
        },
        "dwconv": {
            "model": "/Users/kalineid/test_models/BF_reshape_dwconv_dwconv",
            "shapes": [
                [
                    16,
                    28,
                    28
                ]
            ]
        },
        "block": {
            "model": "/Users/kalineid/test_models/BF_reshape_dwconv_block",
            "shapes": [
                [
                    28,
                    28,
                    16
                ]
            ]
        }
    },
    "BF_reshape_relu": {
        "reshape": {
            "model": "/Users/kalineid/test_models/BF_reshape_relu_reshape",
            "shapes": [
                [
                    28,
                    28,
                    16
                ]
            ]
        },
        "relu": {
            "model": "/Users/kalineid/test_models/BF_reshape_relu_relu",
            "shapes": [
                [
                    16,
                    28,
                    28
                ]
            ]
        },
        "block": {
            "model": "/Users/kalineid/test_models/BF_reshape_relu_block",
            "shapes": [
                [
                    28,
                    28,
                    16
                ]
            ]
        }
    }
}
```

`BF_reshape_dwconv` is the name of a rule. It consists of several test models to profile. Here, there are three models called `reshape`, `dwconv` and `block`. For each model, the `model` field is the path to where the model is saved. `shapes` is its inputs. For example, here `[[28, 28, 16]]` means this model has only one input, and the shape is `(28, 28, 16)`.
