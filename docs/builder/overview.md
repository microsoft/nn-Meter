# nn-Meter Builder

nn-Meter builder is an open source tool for users to build latency predictor on their own devices. There are three main parts in nn-Meter builder:

- **backend**: the module of connecting backends;

- **backend_meta**: the meta tools related to backend, such as fusion rule tester to detect fusion rules for users' backend;

- **kernel_predictor_builder**: the tool to build different kernel latency predictors.

## <span id="create-workspace"> Create Workspace </span>

Before using nn-Meter builder, a workspace folder should be created. In nn-Meter builder, a workspace refers to a direction to save experiment configs, test case models for detecting fusion rules, kernel models for building latency predictor, and results files for a group of experiments. Users can create a workspace folder by running the following command:

``` Bash
# for TFLite platform
nn-meter create --tflite-workspace <path/to/place/workspace/>

# for OpenVINO platform
nn-meter create --openvino-workspace <path/to/place/workspace/>

# for customized platform
nn-meter create --customized-workspace <path/to/place/workspace/> --backend <backend-name>
```

After running the command, a workspace folder will be created, and a series of configuration file will be placed in `<workspace-path>/configs/`. Users could open `<workspace-path>/configs/*.yaml` and edit the content to change configuration. 

Then, users are required to initialize workspace in `builder_config` module via python binding:

```python
from nn_meter.builder import builder_config

# initialize builder config with workspace
builder_config.init(
    workspace_path="path/to/workspace/folder"
) # change the text to required platform type and workspace path
```

Note: after executing ``builder_config.init``, the config are loaded permanently. If users want to update a config, it's required to repeat this initialization process again.

## Connect Backend

Please refer to [prepare_backend.md](prepare_backend.md) to prepare your own backend.

## Detect Fusion Rule

Please refer to [test_fusion_rules.md](test_fusion_rules.md) to detect fusion rule.

## Build Kernel Latency Predictor

Please refer to [build_kernel_latency_predictor.md](build_kernel_latency_predictor.md) to build kernel latency predictor.

## Use Customized Predictor for Latency Prediction
Please refer to [customize_predictor.md](customize_predictor.md) to utilize customized kernel latency predictors for model latency prediction.
