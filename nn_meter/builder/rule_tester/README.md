# Android Guide

## Prepare Android Device

Please follow the [tensorflow official guide](https://www.tensorflow.org/lite/performance/measurement) to build and deploy `benchmark_model` onto the device. 

1. use bazel to build a benchmark_model file. Please refer to https://dev.azure.com/v-wjiany/Personal/_git/tensorflow_modified?path=/tensorflow/lite/tools/benchmark/android/README.md&_a=preview
Follow the tensorflow official guide for a more detailed guide to build and deploy benchmark_model onto the device.
``` Bash
bazel build -c opt //tensorflow/lite/tools/benchmark:benchmark_model
```

2. Push the benchmark_model to edge device by specifing its serial.
``` Bash
adb -s <device-serial> push bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model /data/local/tmp # send to edge
adb shell chmod +x /data/local/tmp/benchmark_model
```

## Config RuleTest
TODO: @ Jiahang: where is config.py?
Directly modify `config.py` or provide a `local_config.py` to overide configurations.

### Required Configs

You need to provide settings for the `PARAMS` field.

- `MODEL_DIR`: Path to the folder (on host device) where temporary models will be generated.
- `REMOTE_MODEL_DIR`: Path to the folder (on android device) where temporary models will be copied to.
- `KERNEL_PATH`: Path (on android device) where the kernel implementations will be dumped.
- `BENCHMARK_MODEL_PATH`: Path (on android device) where `benchmark_model` is deployed.
- `DEVICE_SERIAL`: If there are multiple adb devices connected to your host, you need to provide the corresponding serial id. Leave to `''` if not.

```python
BACKENDS = {
    'tflite_gpu': {
        'ENGINE': 'backends.tflite_gpu',
        'PARAMS': {
            'MODEL_DIR': '/data1/datasets_pad', #os.path.join(HOME_DIR, "benchmarks/models/tflite"),
            'REMOTE_MODEL_DIR': '/mnt/sdcard/tflite_bench',
            'KERNEL_PATH': '/mnt/sdcard/tflite_bench/kernel.cl',
            'BENCHMARK_MODEL_PATH': '/data/local/tmp/benchmark_model',
            'DEVICE_SERIAL': '5e6fecf',
        },
        'ENABLED': True,
    },
}
```

`OUTPUT_PATH` also need to be provided. The generated rule files will be placed there.

### Optional Configs

- `DEFAULT_INPUT_SHAPE`: default resolution and channels of your input image
- `D1_INPUT_SHAPE`: input shapes of 1d operations like `dense`
- `FILTERS`: filter numbers of conv and dwconv
- `KERNEL_SIZE`: kernel size
- `ENABLED`: rules to be tested
- `DETAIL`: whether or not dump the detail inference time in rule files

## Run

```
python main.py
```
