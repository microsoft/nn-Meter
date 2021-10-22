Note: This is an alpha (preview) version which is still under refining.

**nn-Meter** is a novel and efficient system to accurately predict the inference latency of DNN models on diverse edge devices. The key idea is dividing a whole model inference into kernels, i.e., the execution units of fused operators on a device, and conduct kernel-level prediction. We currently evaluate four popular platforms on a large dataset of 26k models. It achieves 99.0% (mobile CPU), 99.1% (mobile Adreno 640 GPU), 99.0% (mobile Adreno 630 GPU), and 83.4% (Intel VPU) prediction accuracy.

The current supported hardware and inference frameworks:

|       Device       |   Framework   |   Processor   | +-10%  Accuracy |      Hardware name      |
| :-----------------: | :------------: | :------------: | :-------------: | :----------------------: |
|       Pixel4       |  TFLite v2.1  | CortexA76 CPU |      99.0%      |  cortexA76cpu_tflite21  |
|         Mi9         |  TFLite v2.1  | Adreno 640 GPU |      99.1%      |  adreno640gpu_tflite21  |
|      Pixel3XL      |  TFLite v2.1  | Adreno 630 GPU |      99.0%      |  adreno630gpu_tflite21  |
| Intel Movidius NCS2 | OpenVINO2019R2 |   Myriad VPU   |      83.4%      | myriadvpu_openvino2019r2 |

*nn-Meter has achieved the **Mobisys 21 Best Paper Award!** For more details, please check out paper:*

[nn-Meter: towards accurate latency prediction of deep-learning model inference on diverse edge devices](https://dl.acm.org/doi/10.1145/3458864.3467882)

## Who should consider using nn-Meter

- Those who want to get the DNN inference latency on mobile and edge devices with **no deployment efforts on real devices**.
- Those who want to run **hardware-aware NAS with [NNI](https://github.com/microsoft/nni)**.
- Those who want to **build latency predictors for their own devices**.
- Those who want to use the 26k latency [benchmark dataset](https://github.com/microsoft/nn-Meter/releases/download/v1.0-data/datasets.zip).

# Installation

Currently nn-Meter has been tested on Linux and Windows system. Windows 10, Ubuntu 16.04 and 20.04 with python 3.6.10 are tested and supported. Please first install `python3` before nn-Meter installation. Then nn-Meter could be installed by running:

```Bash
pip install nn-meter
```

If you want to try latest code, please install nn-Meter from source code. First git clone nn-Meter package to local:

```Bash
git clone git@github.com:microsoft/nn-Meter.git
cd nn-Meter
```

Then simply run the following pip install in an environment that has `python >= 3.6`. The command will complete the automatic installation of all necessary dependencies and nn-Meter.

```Bash
pip install .
```

nn-Meter is a latency predictor of models with type of Tensorflow, PyTorch, Onnx, nn-meter IR graph and [NNI IR graph](https://github.com/microsoft/nni). To use nn-Meter for specific model type, you also need to install corresponding required packages. The well tested versions are listed below:

| Testing Model Type |                                                       Requirements                                                       |
| :----------------: | :-----------------------------------------------------------------------------------------------------------------------: |
|     Tensorflow     |                                                  `tensorflow==1.15.0`                                                  |
|       Torch       | `torch==1.9.0`, `torchvision==0.10.0`, (alternative)[`onnx==1.9.0`, `onnx-simplifier==0.3.6`] or [`nni>=2.4`][1] |
|        Onnx        |                                                      `onnx==1.9.0`                                                      |
| nn-Meter IR graph |                                                            ---                                                            |
|    NNI IR graph    |                                                       `nni>=2.4`                                                       |

[1] Please refer to [nn-Meter Usage](#torch-model-converters) for more information.

Please also check the versions of `numpy` and `scikit_learn`. The different versions may change the prediction accuracy of kernel predictors.

The stable version of wheel binary package will be released soon.

# Usage

To apply for hardware latency prediction, nn-Meter provides two types of interfaces：

- command line `nn-meter` after `nn-meter`[installation](QuickStart.md#Installation).
- Python binding provided by the module `nn_meter`

Here is a summary of supported inputs of the two methods.

| Testing Model Type |                                       Command Support                                       |                                          Python Binding                                          |
| :----------------: | :-----------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------: |
|     Tensorflow     |             Checkpoint file dumped by `tf.saved_model()` and end with `.pb`             |                 Checkpoint file dumped by `tf.saved_model` and end with `.pb`                 |
|       Torch       |                              Models in `torchvision.models`                              |                                   Object of `torch.nn.Module`                                   |
|        Onnx        | Checkpoint file dumped by `torch.onnx.export()` or `onnx.save()` and end with `.onnx` |           Checkpoint file dumped by `onnx.save()` or model loaded by `onnx.load()`           |
| nn-Meter IR graph |     Json file in the format of [nn-Meter IR Graph](./docs/input_models.md#nnmeter-ir-graph)     | `dict` object following the format of [nn-Meter IR Graph](./docs/input_models.md#nnmeter-ir-graph) |
|    NNI IR graph    |                                              -                                              |                                        NNI IR graph object                                        |

In both methods, users could appoint predictor name and version to target a specific hardware platform (device). Currently, nn-Meter supports prediction on the following four configs:

| Predictor (device_inferenceframework) | Processor Category | Version |
| :-----------------------------------: | :----------------: | :-----: |
|         cortexA76cpu_tflite21         |        CPU        |   1.0   |
|         adreno640gpu_tflite21         |        GPU        |   1.0   |
|         adreno630gpu_tflite21         |        GPU        |   1.0   |
|       myriadvpu_openvino2019r2       |        VPU        |   1.0   |

Users can get all predefined predictors and versions by running

```bash
# to list all predefined predictors
nn-meter --list-predictors 
```

## Predict latency of saved CNN model

After installation, a command named `nn-meter` is enabled. To predict the latency for a CNN model with a predefined predictor in command line, users can run the following commands

```bash
# for Tensorflow (*.pb) file
nn-meter lat_pred --predictor <hardware> [--predictor-version <version>] --tensorflow <pb-file_or_folder> 

# for ONNX (*.onnx) file
nn-meter lat_pred --predictor <hardware> [--predictor-version <version>] --onnx <onnx-file_or_folder>

# for torch model from torchvision model zoo (str)
nn-meter lat_pred --predictor <hardware> [--predictor-version <version>] --torchvision <model-name> <model-name>... 

# for nn-Meter IR (*.json) file
nn-meter lat_pred --predictor <hardware> [--predictor-version <version>] --nn-meter-ir <json-file_or_folder> 
```

`--predictor-version <version>` arguments is optional. When the predictor version is not specified by users, nn-meter will use the latest version of the predictor.

nn-Meter can support batch mode prediction. To predict latency for multiple models in the same model type once, user should collect all models in one folder and state the folder after `--[model-type]` liked argument.

It should also be noted that for PyTorch model, nn-meter can only support existing models in torchvision model zoo. The string followed by `--torchvision` should be exactly one or more string indicating name(s) of some existing torchvision models. To apply latency prediction for torchvision model in command line, `onnx` and `onnx-simplifier` packages are required.

### Convert to nn-Meter IR Graph

Furthermore, users may be interested to convert tensorflow pb-file or onnx file to nn-Meter IR graph. Users could convert nn-Meter IR graph and save to `.json` file be running

```bash
# for Tensorflow (*.pb) file
nn-meter get_ir --tensorflow <pb-file> [--output <output-name>]

# for ONNX (*.onnx) file
nn-meter get_ir --onnx <onnx-file> [--output <output-name>]
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

In `predictor.predict()`, the allowed items of the parameter `model_type` include `["pb", "torch", "onnx", "nnmeter-ir", "nni-ir"]`, representing model types of tensorflow, torch, onnx, nn-meter IR graph and NNI IR graph, respectively.

`<span id="torch-model-converters">` For Torch models, the shape of feature maps is unknown merely based on the given network structure, which is, however, significant parameters in latency prediction. Therefore, torch model requires a shape of input tensor for inference as a input of `predictor.predict()`. Based on the given input shape, a random tensor according to the shape will be generated and used. Another thing for Torch model prediction is that users can install the `onnx` and `onnx-simplifier` packages for latency prediction (referred to as Onnx-based latency prediction for torch model), or alternatively install the `nni` package (referred to as NNI-based latency prediction for torch model). Note that the `nni` option does not support command line calls. In addition, if users use `nni` for latency prediction, the PyTorch modules should be defined by the `nn` interface from NNI `import nni.retiarii.nn.pytorch as nn` (view [NNI doc](https://nni.readthedocs.io/en/stable/NAS/QuickStart.html#define-base-model) for more information), and the parameter `apply_nni` should be set as `True` in the function `predictor.predict()`. Here is an example of NNI-based latency prediction for Torch model:

```python
import nni.retiarii.nn.pytorch as nn
from nn_meter import load_latency_predictor

predictor = load_latency_predictor(...)

# build your model using nni.retiarii.nn.pytorch as nn
model = nn.Module ...

input_shape = (1, 3, 224, 224)
lat = predictor.predict(model, model_type='torch', input_shape=input_shape, apply_nni=True) 
```

The Onnx-based latency prediction for torch model is stable but slower, while the NNI-based latency prediction for torch model is unstable as it could fail in some case but much faster compared to the Onnx-based model. The Onnx-based model is set as the default one for Torch model latency prediction in nn-Meter. Users could choose which one they preferred to use according to their needs. 

Users could view the information all built-in predictors by `list_latency_predictors` or view the config file in `nn_meter/configs/predictors.yaml`.

Users could get a nn-Meter IR graph by applying `model_file_to_graph` and `model_to_graph` by calling the model name or model object and specify the model type. The supporting model types of `model_file_to_graph` include "onnx", "pb", "torch", "nnmeter-ir" and "nni-ir", while the supporting model types of `model_to_graph` include "onnx", "torch" and "nni-ir".

## Benchmark Dataset

To evaluate the effectiveness of a prediction model on an arbitrary DNN model, we need a representative dataset that covers a large prediction scope. As there is no such available latency dataset, nn-Meter collects and generates 26k CNN models. It contains various operators, configurations, and edge connections, with covering different levels of FLOPs and latency. (Please refer the paper for the dataset generation method and dataset numbers.)

We release the dataset, and provide an interface of `nn_meter.dataset` for users to get access to the dataset. Users can also download the data from the [Download Link](https://github.com/microsoft/nn-Meter/releases/download/v1.0-data/datasets.zip) for testing nn-Meter or their own prediction models. 

## Hardware-aware NAS by nn-Meter and NNI

To empower affordable DNN on the edge and mobile devices, hardware-aware NAS searches both high accuracy and low latency models. In particular, the search algorithm only considers the models within the target latency constraints during the search process.

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

## Bench Dataset

To evaluate the effectiveness of a prediction model on an arbitrary DNN model, we need a representative dataset that covers a large prediction scope. nn-Meter collects and generates 26k CNN models. (Please refer the paper for the dataset generation method.)

We release the dataset, and provide an interface of `nn_meter.dataset` for users to get access to the dataset. Users can also download the data from the [Download Link](https://github.com/microsoft/nn-Meter/releases/download/v1.0-data/datasets.zip) on their own. 



# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# License

The entire codebase is under [MIT license](https://github.com/microsoft/nn-Meter/blob/main/LICENSE)

The dataset is under [Open Use of Data Agreement](https://github.com/Community-Data-License-Agreements/Releases/blob/main/O-UDA-1.0.md)

# Citation

If you find that nn-Meter helps your research, please consider citing it:

```
@inproceedings{nnmeter,
    author = {Zhang, Li Lyna and Han, Shihao and Wei, Jianyu and Zheng, Ningxin and Cao, Ting and Yang, Yuqing and Liu, Yunxin},
    title = {nn-Meter: Towards Accurate Latency Prediction of Deep-Learning Model Inference on Diverse Edge Devices},
    year = {2021},
    publisher = {ACM},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3458864.3467882},
    doi = {10.1145/3458864.3467882},
    booktitle = {Proceedings of the 19th Annual International Conference on Mobile Systems, Applications, and Services},
    pages = {81–93},
}

@misc{nnmetercode,
    author = {Microsoft Research nn-Meter Team},
    title = {nn-Meter: Towards Accurate Latency Prediction of Deep-Learning Model Inference on Diverse Edge Devices},
    year = {2021},
    url = {https://github.com/microsoft/nn-Meter},
}
```
