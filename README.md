# Introduction

>nn-Meter is a novel and efficient system to accurately predict the inference latency of DNN models on diverse edge devices. The key idea is dividing a whole model inference into kernels, i.e., the execution units of fused operators on a device, and conduct kernel-level prediction. 
nn-Meter contains two key techniques: (i) kernel detection to automatically detect the execution unit of model inference via a set of well-designed test cases; (ii) adaptive sampling to efficiently sample the most beneficial configurations from a large space to build accurate kernel-level latency predictors.
nn-Meter currently evaluates four popular platforms on a large dataset of 26k models. It achieves 99.0% (mobile CPU), 99.1% (mobile Adreno 640 GPU), 99.0% (mobile Adreno 630 GPU), and 83.4% (Intel VPU) prediction accuracy.

The current supported hardware and inference frameworks:

| name |        Device       |    Framework   |    Processor   | +-10%  Accuracy |
|:----:|:-------------------:|:--------------:|:--------------:|:------------------:|
|  CPU |        Pixel4       |   TFLite v2.1  |  CortexA76 CPU |        99.0%       |
|  GPU |         Mi9         |   TFLite v2.1  | Adreno 640 GPU |        99.1%       |
| GPU1 |       Pixel3XL      |   TFLite v2.1  | Adreno 630 GPU |        99.0%       |
|  VPU | Intel Movidius NCS2 | OpenVINO2019R2 |   Myriad VPU   |        83.4%       |

As the maintainer of this project, please make a few updates:

- Improving this README.MD file to provide a great experience
- Updating SUPPORT.MD with content about this project's support experience
- Understanding the security reporting process in SECURITY.MD
- Remove this section from the README

## Installation

To install nn-meter, please first install python3. The test environment uses anaconda python 3.6.10. Install the dependencies via: 
`pip3 install -r requirements.txt`
Please also check the versions of numpy, scikit_learn. The different versions may change the prediction accuracy of kernel predictors. 

## Usage

### Run nn-Meter demo
To predict the latency for a CNN model on a hardware, users can run the following command with two hyper-parameters:
```
python demo.py --hardware cortexA76cpu_tflite21 --input_model data/testmodels/alexnet_0.pb
```
The two hyper-parameters include: (i) the targeting hardware with inference framework, (ii) the input model file

nn-Meter currently supports prediction on the following four hardware:
|  hardware | 
|:-------------------:|
|cortexA76cpu_tflite21 | 
|adreno640gpu_tflite21 |
|adreno630gpu_tflite21  |
|myriadvpu_openvino2019r2 |

For the input model file, you can find any example provided under the `data/testmodels`




### Predict inference latency (to do)
nn-Meter could be seamlessly integrated with existing `PyTorch` codes to predict the inference latency of an `torch.nn.Module` object. 
```python
from nn_meter import load_latency_predictor

predictor = load_lat_predictor(backend='TFLite-CortexA76') # case insensitive in backend

# build your model here
model = ... # model is instance of torch.nn.Module

lat = predictor.predict(model)
```
By calling `load_latency_predictor`, user selects the target backend (`Framework-Hardware`) and loads the corresponding predictor. nn-Meter will try to find the right predictor file in `~/.nn_meter/predictors`. If the predictor file doesn't exist, it will download from the Github repo. 

Users could view the information all built-in predictors by `list_latency_predictors` or view the config file in `~/.nn_meter/config.json`.

### Use nn-Meter in commands (to do)
To predict the latency for saved models, users could also use the nn-Meter command like 

```bash
nn-meter --input_model data/testmodels/alexnet.onnx --backend TFLite-CortexA76
```
Currently we support `ONNX` format (ONNX files of popular CNN models are included in [`data/testmodels`](data/testmodels)) and Tensorflow pb file. 

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## License
The entire codebase is under [MIT license](https://github.com/microsoft/nn-Meter/blob/main/LICENSE) 

The dataset is under [Open Use of Data Agreement](https://github.com/Community-Data-License-Agreements/Releases/blob/main/O-UDA-1.0.md)

