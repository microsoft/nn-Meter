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

To run the latency predictor, we support two input formats. We include popular CNN models in `data/testmodels`

#### 1. input model: xx.onnx or xx.pb :

`python demo.py --input_model data/testmodels/alexnet.onnx --mf alexnet`

`python demo.py --input_model data/testmodels/alexnet.pb --mf alexnet`

It will firstly convert onnx and pb models into our defined IR json. We conduct kernel detection with the IR graph and predict kernel latency on the 4 measured edge devices. 

#### 2. input model: the converted IR json:

`python demo.py --input_models data/testmodels/alexnet_0.json --mf alexnet`

#### 3. To convert the onnx and pb model into the IR json:

`python model_converter.py --input_model data/testmodels/alexnet_0.pb --output_path data/testmodels/alexnet_0.json`


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

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
