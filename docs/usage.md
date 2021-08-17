# Usage

To apply for hardware latency prediction, nn-Meter provides two types of interfaces：

- command line `nn-meter` after `nn-meter` [installation](QuickStart.md#Installation).
- Python binding provided by the module `nn_meter`

Here is a summary of supported inputs of the two methods.

|       Testing Model Type       |                                   Command Support                                   |                                                   Python Binding                                                   |
| :---------------: | :---------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------: |
|    Tensorflow    |         Checkpoint file dumped by `tf.saved_model()` and endwith `.pb`         |                          Checkpoint file dumped by `tf.saved_model` and endwith `.pb`                          |
|       Torch       |                          Models in `torchvision.models`                          |                                            Object of `torch.nn.Module`                                            |
|       Onnx       |           Checkpoint file dumped by `onnx.save()` and endwith `.onnx`           |                    Checkpoint file dumped by `onnx.save()` or model loaded by `onnx.load()`                    |
| nn-Meter IR graph | Json file in the format of [nn-Meter IR Graph](./docs/input_models.md#nnmeter-ir-graph) |          `dict` object following the format of [nn-Meter IR Graph](./docs/input_models.md#nnmeter-ir-graph)          |
|   NNI IR graph   |                                          -                                          | NNI IR graph object |

In both methods, users could appoint predictor name and version to target a specific hardware platform (device). Currently, nn-Meter supports prediction on the following four configs:
| Predictor (device_inferenceframework) | Processor Category | Version |
| :-----------------------------------: | :----------------: | :-----: |
|         cortexA76cpu_tflite21         |        CPU         |   1.0   |
|         adreno640gpu_tflite21         |        GPU         |   1.0   |
|         adreno630gpu_tflite21         |        GPU         |   1.0   |
|       myriadvpu_openvino2019r2        |        VPU         |   1.0   |

Users can get all predefined predictors and versions by running

```bash
# to list all predefined predictors
nn-meter --list-predictors 
```

## Predict latency of saved CNN model

After installation, a command named `nn-meter` is enabled. To predict the latency for a CNN model with a predefined predictor in command line, users can run the following commands

```bash
# for Tensorflow (*.pb) file
nn-meter --predictor <hardware> [--predictor-version <version>] --tensorflow <pb-file_or_folder> 

# for ONNX (*.onnx) file
nn-meter --predictor <hardware> [--predictor-version <version>] --onnx <onnx-file_or_folder>

# for torch model from torchvision model zoo (str)
nn-meter --predictor <hardware> [--predictor-version <version>] --torchvision <model-name> <model-name>... 

# for nn-Meter IR (*.json) file
nn-meter --predictor <hardware> [--predictor-version <version>] --nn-meter-ir <json-file_or_folder> 
```

`--predictor-version <version>` arguments is optional. When the predictor version is not specified by users, nn-meter will use the latest verison of the predictor.

nn-Meter can support batch mode prediction. To predict latency for multiple models in the same model type once, user should collect all models in one folder and state the folder after `--[model-type]` liked argument.

It should also be noted that for PyTorch model, nn-meter can only support existing models in torchvision model zoo. The string followed by `--torchvision` should be exactly one or more string indicating name(s) of some existing torchvision models.

### Convert to nn-Meter IR Graph

Furthermore, users may be interested to convert tensorflow pb-file or onnx file to nn-Meter IR graph. Users could convert nn-Meter IR graph and save to `.json` file be running

```bash
# for Tensorflow (*.pb) file
nn-meter getir --tensorflow <pb-file> [--output <output-name>]

# for ONNX (*.onnx) file
nn-meter getir --onnx <onnx-file> [--output <output-name>]
```

Output name is default to be `/path/to/input/file/<input_file_name>_<model-type>_ir.json` if not specified by users.

## Use nn-Meter in your python code

After installation, users can import nn-Meter in python code

```python
from nn_meter import load_latency_predictor

predictor = load_latency_predictor(hardware_name, hardware_predictor_version) # case insensitive in backend

# build your model (e.g., model instance of torch.nn.Module)
model = ... 

lat = predictor.predict(model, model_type) # the resulting latency is in unit of ms
```

By calling `load_latency_predictor`, user selects the target hardware and loads the corresponding predictor. nn-Meter will try to find the right predictor file in `~/.nn_meter/data`. If the predictor file doesn't exist, it will download from the Github release.

In `predictor.predict`, the allowed items of the parameter `model_type` include `["pb", "torch", "onnx", "nnmeter-ir", "nni-ir"]`, representing model types of tensorflow, torch, onnx, nn-meter IR graph and NNI IR graph, respectively.

Users could view the information all built-in predictors by `list_latency_predictors` or view the config file in `nn_meter/configs/predictors.yaml`.

Users could get a nn-Meter IR graph by applying `model_file_to_graph` and `model_to_graph` by calling the model name or model object and specify the model type. The supporting model types of `model_file_to_graph` include "onnx", "pb", "torch", "nnmeter-ir" and "nni-ir", while the supporting model types of `model_to_graph` include "onnx", "torch" and "nni-ir".

## Hardware-aware NAS by nn-Meter and NNI

To empower affordable DNN on the edge and mobile devices, hardware-aware NAS searches both high accuracy and low latency models. In particular, the search algorithm only considers the models within the target latency constraints during the search process. For more theoretical details, please refer to [this doc](hardware-aware-model-design.md).

Currently we provides example of end-to-end [multi-trial NAS](https://nni.readthedocs.io/en/stable/NAS/multi_trial_nas.html), which is a [random search algorithm](https://arxiv.org/abs/1902.07638) on [SPOS NAS](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610528.pdf) search space. More examples of more hardware-aware NAS and model compression algorithms are coming soon. 

To run multi-trail SPOS demo, NNI should be installed through source code by following [NNI Doc](https://nni.readthedocs.io/en/stable/Tutorial/InstallationLinux.html#installation)
```bash
python setup.py develop
```

Then run multi-trail SPOS demo:

```bash
python ${NNI_ROOT}/examples/nas/oneshot/spos/multi_trial.py
```

### How the demo works

Refer to [NNI Doc](https://nni.readthedocs.io/en/stable/nas.html) for how to perform NAS by NNI.

To support hardware-aware NAS, you first need a `Strategy` that supports filtering the models by latency. We provide such a filter named `LatencyFilter` in NNI and initialize a `Random` strategy with the filter:

```python
simple_strategy = strategy.Random(model_filter=LatencyFilter(threshold=100, predictor=base_predictor))
```

`LatencyFilter` will predict the models' latency by using nn-Meter and filter out the models whose latency with the given predictor are larger than the threshold (i.e., `100` in this example).
You can also build your own strategies and filters to support more flexible NAS such as sorting the models according to latency.

Then, pass this strategy to `RetiariiExperiment`:

```python
exp = RetiariiExperiment(base_model, trainer, strategy=simple_strategy)

exp_config = RetiariiExeConfig('local')
...
exp_config.dummy_input = [1, 3, 32, 32]

exp.run(exp_config, port)
```
In `exp_config`, `dummy_input` is required for tracing shape info.