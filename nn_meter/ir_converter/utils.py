# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import logging
from .onnx_converter import OnnxConverter
from .frozenpb_converter import FrozenPbConverter
from .torch_converter import NNIBasedTorchConverter, OnnxBasedTorchConverter, NNIIRConverter
from nn_meter.utils.import_package import try_import_onnx, try_import_torch, try_import_torchvision_models
logging = logging.getLogger("nn-Meter")


def model_file_to_graph(filename: str, model_type: str, input_shape=(1, 3, 224, 224), apply_nni=False):
    """
    read the given file and convert the model in the file content to nn-Meter IR graph object 
    @params:
    filename: string to specify the location of the file
    
    model: the model to be predicted, allowed file format include
        - the path to a saved tensorflow model file (*.pb), `model_type` must be set to "pb"
        - string to specify the name of a built-in torch model from the torchvision model zoo, `model_type` must be set to "torch"
        - the path to a saved ONNX model file (*.onnx), `model_type` must be set to "onnx"
        - the path to a saved dictionary object following nn-Meter-IR format (*.json), `model_type` must be set to "nnmeter-ir"
        - the path to a saved dictionary object following NNI-IR format(*.json), `model_type` must be set to "nni-ir"
        
    model_type:  string to specify the type of parameter model, allowed items are ["pb", "torch", "onnx", "nnmeter-ir", "nni-ir"]
    
    input_shape: the shape of input tensor for inference (if necessary), a random tensor according to the shape will be generated and used. This parameter is only 
        accessed when model_type == 'torch'
    
    apply_nni: switch the torch converter used for torch model parsing. If apply_nni==True, NNI-based converter is used for torch model conversion, which requires 
        nni>=2.4 installation and should use nn interface from NNI `import nni.retiarii.nn.pytorch as nn` to define the PyTorch modules. Otherwise Onnx-based torch 
        converter is used, which requires onnx installation (well tested version is onnx==1.9.0). NNI-based converter is much faster while the conversion is unstable 
        as it could fail in some case. Onnx-based converter is much slower but stable compared to NNI-based converter. This parameter is only accessed when 
        model_type == 'torch'
    """
    if model_type == "onnx":
        onnx = try_import_onnx()
        model = onnx.load(filename)
        return onnx_model_to_graph(model)

    elif model_type == "pb":
        converter = FrozenPbConverter(filename)
        return converter.get_flatten_graph()

    elif model_type == "nni-ir":
        with open(filename, "r") as fp:
            model = json.load(fp)
        return nni_model_to_graph(model)

    elif model_type == "nnmeter-ir":
        with open(filename, "r") as fp:
            return json.load(fp)

    elif model_type == "torch":
        models = try_import_torchvision_models()
        torchvision_zoo_dict = {
            'resnet18': 'models.resnet18()',
            'alexnet': 'models.alexnet()',
            'vgg16': 'models.vgg16()',
            'squeezenet': 'models.squeezenet1_0()',
            'densenet161': 'models.densenet161()',
            'inception_v3': 'models.inception_v3()',
            'googlenet': 'models.googlenet()',
            'shufflenet_v2': 'models.shufflenet_v2_x1_0()',
            'mobilenet_v2': 'models.mobilenet_v2()',
            'resnext50_32x4d': 'models.resnext50_32x4d()',
            'wide_resnet50_2': 'models.wide_resnet50_2()',
            'mnasnet': 'models.mnasnet1_0()',
        }
        if filename in torchvision_zoo_dict:
            model = eval(torchvision_zoo_dict[filename])
        else:
            suppost_list = ", ".join([k for k in torchvision_zoo_dict])
            raise ValueError(f"Unsupported model name: {filename} in torchvision. Supporting list: {suppost_list}")
        return torch_model_to_graph(model, input_shape, apply_nni)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def model_to_graph(model, model_type, input_shape=(1, 3, 224, 224), apply_nni=False):
    """
    convert the given model to nn-Meter IR graph object 
    @params:
    model: the model object for converting, allowed file format include
        - pytorch model object (nn.Module), `model_type` must be set to "torch"
        - ONNX model object, `model_type` must be set to "onnx"
        - dictionary object following NNI-IR format, `model_type` must be set to "nni-ir"
        
    model_type:  string to specify the type of parameter model, allowed items are ["torch", "onnx", "nnmeter-ir", "nni-ir"]
    
    input_shape: the shape of input tensor for inference (if necessary), a random tensor according to the shape will be generated and used. This parameter is only 
        accessed when model_type == 'torch'
    """
    if model_type == "onnx":
        return onnx_model_to_graph(model)
    elif model_type == "torch":
        return torch_model_to_graph(model, input_shape, apply_nni)
    elif model_type == "nni-ir":
        return nni_model_to_graph(model)
    elif model_type == "nnmeter-ir":
        return model # nnmeter-ir doesn't need any post-process
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def onnx_model_to_graph(model):
    converter = OnnxConverter(model)
    return converter.convert()


def nni_model_to_graph(model):
    converter = NNIIRConverter(model)
    return converter.convert()


def torch_model_to_graph(model, input_shape=(1, 3, 224, 224), apply_nni=False):
    torch = try_import_torch()
    args = torch.randn(*input_shape)
    try:
        # if the test model has no parameters (such as activation ops), there will be error when calling ``model.parameters``
        if next(model.parameters()).is_cuda:
            args = args.to("cuda")
    except:
        pass
    if apply_nni: 
        # apply NNI-based torch converter, which requires nni>=2.4 installation and should use nn interface from NNI 
        # `import nni.retiarii.nn.pytorch as nn` to define the PyTorch modules.
        try:
            logging.info("NNI-based Torch Converter is applied for model conversion")
            converter = NNIBasedTorchConverter(model, args)
        except:
            raise NotImplementedError("Your model is not fully converted by NNI-based converter. Please set apply_nni=False and try again.")
    else:
        # apply Onnx-based torch converter, which requires onnx installation (well tested version is onnx==1.9.0) 
        # and the conversion is more stable
        logging.info("Onnx-based Torch Converter is applied for model conversion")
        converter = OnnxBasedTorchConverter(model, args)
    return converter.convert()
