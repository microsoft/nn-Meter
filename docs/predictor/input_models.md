# Input Models

nn-Meter currently supports both a saved model file and the model object in code. In particular, we support saved models in .pb and .onnx formats, and we support to directly predict Onnx and PyTorch models. Besides, to support the hardware-aware NAS, nn-Meter can also predict the inference latency of models in [NNI graph](https://nni.readthedocs.io/en/stable/nas.html).

When taking the different model formats as input, nn-Meter converts them in nn-Meter IR graph. The kernel detection code will split the nn-Meter IR graph into the set of kernel units, and conduct kernel-level prediction.

### Input model as a saved file

You can save tensorflow models into frozen pb formats, and use the following nn-meter command to predict the latency:

```bash
# for Tensorflow (*.pb) file
nn-meter predict --predictor <hardware> [--predictor-version <version>] --tensorflow <pb-file_or_folder> 
```

For the other frameworks (e.g., PyTorch), you can convert the models into onnx models, and use the following nn-meter command to predict the latency:

```bash
# for ONNX (*.onnx) file
nn-meter predict --predictor <hardware> [--predictor-version <version>] --onnx <onnx-file_or_folder>
```

You can download the test [tensorflow models]("https://github.com/microsoft/nn-Meter/releases/download/v1.0-data/pb_models.zip") and [onnx models](https://github.com/microsoft/nn-Meter/releases/download/v1.0-data/onnx_models.zip). 

### Input model as a code object

You can also directly apply nn-Meter in your python code. In this case, please directly pass the onnx model and PyTorch model objects as the input model. The following is an example for PyTorch code:

```python
from nn_meter import load_latency_predictor

predictor = load_latency_predictor(hardware_name) # case insensitive in backend

# build your model here
model = ... # model is instance of torch.nn.Module

lat = predictor.predict(model, model_type='torch', input_shape=(3, 224, 224), apply_nni=False)
```

There are two converters, i.e., model processors, for torch model, namely the Onnx-based torch converter and the NNI-based torch converter. Onnx-based torch converter export the torch model to onnx model, and reload the onnx model to the onnx converter. The serialization and postprocessing for Onnx-based torch converter is time-consuming, but the Onnx conversion is more stable. 

NNI-based torch converter generate a NNI IR graph based on the torch model, and use NNI converter for the subsequent steps. Note that if users use NNI-based converter, the PyTorch modules should be defined by the `nn` interface from NNI `import nni.retiarii.nn.pytorch as nn` (view [NNI doc](https://nni.readthedocs.io/en/stable/NAS/QuickStart.html#define-base-model) for more information). NNI-based torch converter get advantage in speed, but could fail in case the model contains some operators not supported by NNI. 

One can switch two converters by setting `True` or `False` of the parameter `apply_nni` in `predictor.predict()`. Onnx-based torch converter is used as the default one for torch model. If `apply_nni-True`, NNI-based torch converter is used instead. Users could choose which one they preferred to use according to their needs. 

### <span id="nnmeter-ir-graph"> nn-Meter IR graph </span>

As introduced, nn-Meter will perform a pre-processing step to convert the above model formats into the nn-Meter IR graphs. Now we introduce the defined IR graph.

A *model* is consisting of *nodes*. The following is an example of conv *node*  of AlexNet model

<img src="imgs/irgraph.png" alt="drawing" width="400"/>

For a *node*, we use the identical node name ("conv1.conv/Conv2D") as the node key. A *node* consists of:

* inbounds: a list of incoming node names
* outbounds: a list of outgoing node names. The inbounds and outbounds describe the node connections.
* attr: a set of attributes for the node. The attributes can be different for different types of NN node.

You can download the example nn-Meter IR graphs through [here](https://github.com/microsoft/nn-Meter/releases/download/v1.0-data/ir_graphs.zip).

When you have a large amount of models to predict, you can also convert them into nn-Meter IR graphs to save the pre-processing time:

```
# for Tensorflow (*.pb) file
nn-meter get_ir --tensorflow <pb-file> [--output <output-name>]

# for ONNX (*.onnx) file
nn-meter get_ir --onnx <onnx-file> [--output <output-name>]
```
