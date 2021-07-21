
nn-Meter is a novel and efficient system to accurately predict the inference latency of DNN models on diverse edge devices. The key idea is dividing a whole model inference into kernels, i.e., the execution units of fused operators on a device, and conduct kernel-level prediction.
nn-Meter contains two key techniques: (i) kernel detection to automatically detect the execution unit of model inference via a set of well-designed test cases; (ii) adaptive sampling to efficiently sample the most beneficial configurations from a large space to build accurate kernel-level latency predictors.
We currently evaluate four popular platforms on a large dataset of 26k models. It achieves 99.0% (mobile CPU), 99.1% (mobile Adreno 640 GPU), 99.0% (mobile Adreno 630 GPU), and 83.4% (Intel VPU) prediction accuracy.

The current supported hardware and inference frameworks:

| Abbr. |       Device        |   Framework    |   Processor    | +-10%  Accuracy |    key in nn-Meter usage     |
| :---: | :-----------------: | :------------: | :------------: | :-------------: | :--------------------------: |
|  CPU  |       Pixel4        |  TFLite v2.1   | CortexA76 CPU  |      99.0%      |  **cortexA76cpu_tflite21**   |
|  GPU  |         Mi9         |  TFLite v2.1   | Adreno 640 GPU |      99.1%      |  **adreno640gpu_tflite21**   |
| GPU1  |      Pixel3XL       |  TFLite v2.1   | Adreno 630 GPU |      99.0%      |  **adreno630gpu_tflite21**   |
|  VPU  | Intel Movidius NCS2 | OpenVINO2019R2 |   Myriad VPU   |      83.4%      | **myriadvpu_openvino2019r2** |


## Who should consider using nn-Meter
- Those who want to get the DNN inference latency on mobile and edge devices with **no deployment efforts on real devices**.
- Those who want to run **hardware-aware NAS with [NNI](https://github.com/microsoft/nni)**.
- Those who want to **build latency predictors for their own devices**.
## Installation

To install nn-meter, please first install python3. The test environment uses anaconda python 3.6.10. Install the dependencies via:
`pip3 install -r requirements.txt`
Please also check the versions of numpy, scikit_learn. The different versions may change the prediction accuracy of kernel predictors.

## Usage
### Supported input model format

|       name       | format |
| :--------------: | :----: |
|    Tensorflow    |  .pb   |
|       Onnx       | .onnx  |
| nnmeter IR graph | .json  |
|   NNI IR graph   | .json  |
### To predict a single model: Run nn-Meter demo
After installation, a command named `nn-meter` is enabled. To predict the latency for a CNN model with a predefined predictor, users can run the following commands

```bash
# to list all predefined predictors
nn-meter --list-predictors 

# for Tensorflow (*.pb) file
nn-meter --predictor <hardware> --tensorflow <pb-file> 

# for ONNX (*.onnx) file
nn-meter --predictor <hardware> --onnx <onnx-file>

# for nn-Meter IR (*.json) file
nn-meter --predictor <hardware> --nn-meter-ir <json-file> 

# for NNI IR (*.json) file
nn-meter --predictor <hardware> --nni-ir <json-file> 
```

nn-Meter currently supports prediction on the following four config:

|   Predictor (hardware)   |
| :----------------------: |
|  cortexA76cpu_tflite21   |
|  adreno640gpu_tflite21   |
|  adreno630gpu_tflite21   |
| myriadvpu_openvino2019r2 |

For the input model file, you can find any example provided under the `data/testmodels`


### Import nn-Meter in your python code
```python
from nn_meter import load_latency_predictor

predictor = load_lat_predictor(config, hardware_name) # case insensitive in backend

# build your model here
model = ... # model is instance of torch.nn.Module

lat = predictor.predict(model)
```
By calling `load_latency_predictor`, user selects the target hardware (`Framework-Hardware`) and loads the corresponding predictor. nn-Meter will try to find the right predictor file in `~/.nn_meter/data`. If the predictor file doesn't exist, it will download from the Github release.

Users could view the information all built-in predictors by `list_latency_predictors` or view the config file in `nn_meter/configs/predictors.yaml`.



### Hardware-aware NAS by nn-Meter and NNI

#### Run multi-trial SPOS demo
Install NNI by following [NNI Doc](https://nni.readthedocs.io/en/stable/Tutorial/InstallationLinux.html#installation).

Install nn-Meter from source code (currently we haven't released this package, so development installation is required).

```bash
python setup.py develop
```

Then run multi-trail SPOS demo:

```bash
python ${NNI_ROOT}/examples/nas/oneshot/spos/multi_trial.py
```

#### How the demo works
Refer to https://nni.readthedocs.io/en/stable/nas.html for how to perform NAS by NNI.

To support latency-aware NAS, you first need a `Strategy` that supports filtering the models by latency. We provide such a filter named `LatencyFilter` in NNI and initialize a `Random` strategy with the filter:

```python
simple_strategy = strategy.Random(model_filter=LatencyFilter(100)
```

`LatencyFilter` will predict the models' latency by using nn-Meter and filter out the models whose latency are larger than the threshold (i.e., `100` in this example).
You can also build your own strategies and filters to support more flexible NAS such as sorting the models according to latency.

Then, pass this strategy to `RetiariiExperiment` along with some additional arguments: `parse_shape=True, example_inputs=example_inputs`:
```python
RetiariiExperiment(base_model, trainer, [], simple_strategy, True, example_inputs)
```
Here, `parse_shape=True` means extracting shape info from the torch model as it is required by nn-Meter to predict latency. `example_inputs` is required for tracing shape info.

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

