# Quick Start

## nn-Meter Installation

Currently, nn-Meter has been tested on Linux and Windows system. Windows 10, Ubuntu 16.04 and 20.04 with python 3.6.10 are tested and supported. Please first install `python3` before nn-Meter installation.

We haven't released this package yet, so development installation is required. To install the latest version of nn-Meter, users should install the package through source code. First git clone nn-Meter package to local:
```Bash
git clone git@github.com:microsoft/nn-Meter.git
cd nn-Meter
```
Then simply run the following pip install in an environment that has `python >= 3.6`. The command will complete the automatic installation of all necessary dependencies and nn-Meter.
```Bash
pip install .
```

nn-Meter is a latency predictor of models with type of tensorflow, pytorch, onnx, nn-meter IR graph and [NNI IR graph](https://github.com/microsoft/nni). To use nn-Meter for specific model type, you also need to install corresponding pacakges. The well tested versions are listed below:

|  Testing Model Tpye   |                       Requirments                      |
| :-------------------: | :------------------------------------------------:     |
|       Tensorflow      |  `tensorflow==1.15.0`                                  |
|         Torch         |  `onnx==1.9.0`, `torch==1.7.1`, `torchvision==0.8.2`  |
|          Onnx         |  `onnx==1.9.0`                                         |
|    nn-Meter IR graph  |   ---                                                  |
|      NNI IR graph     |  `nni==2.4`                                            |

Please also check the versions of `numpy` and `scikit_learn`. The different versions may change the prediction accuracy of kernel predictors.

The stable version of wheel binary pacakge will be released soon.


## "Hello World" example on torch model
nn-Meter is an accurate inference latency predictor for DNN models on diverse edge devices. nn-Meter supports tensorflow pb-file, onnx file, torch model and nni IR model for latency prediction.

Here is an example script to predict latency for Resnet18 in torch. To run the example, package `torch`, `torchvision` and `onnx` are required. The well tested versions are `torch==1.7.1`, `torchvision==0.8.2` and `onnx==1.9.0`.   

```python
from nn_meter import load_latency_predictor
import torchvision.models as models

def main():
    base_model = models.resnet18()
    base_predictor = 'cortexA76cpu_tflite21'

    # load a predictor (device_inferenceframework)
    predictor = load_latency_predictor(base_predictor) 

    # predict the latency based on the given model
    lat = predictor.predict(model=base_model, model_type='torch', input_shape=[1, 3, 32, 32]) # in unit of ms
    print(f'Latency for {base_predictor}: {lat} ms')

if __name__ == '__main__':
    main()
```

For more detailed usage of nn-Meter, please refer to [this doc](usage.md).
