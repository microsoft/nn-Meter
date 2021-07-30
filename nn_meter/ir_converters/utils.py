# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
from nn_meter.utils.utils import try_import_onnx, try_import_torch
from .onnx_converter import OnnxConverter
from .frozenpb_converter import FrozenPbConverter
from .torch_converter import TorchConverter
from .torch_converter.converter import NNIIRConverter

def model_file_to_graph(filename, model_type, input_shape=(1, 3, 224, 224)):
    """
    @params:

    input_shape: only accessed when model_type == 'torch'
    """
    if model_type == "onnx":
        onnx = try_import_onnx()
        model = onnx.load(filename)
        return onnx_model_to_graph(model)

    elif model_type == "pb":
        converter = FrozenPbConverter(filename)
        return converter.get_flatten_graph()

    elif model_type == "nni":
        with open(filename, "r") as fp:
            model = json.load(fp)
        return nni_model_to_graph(model)

    elif model_type == "nnmeter":
        with open(filename, "r") as fp:
            return json.load(fp)

    elif model_type == "torch":
        onnx = try_import_onnx()
        model = onnx.load(filename)
        return torch_model_to_graph(model, input_shape)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def onnx_model_to_graph(model):
    converter = OnnxConverter(model)
    return converter.convert()

def nni_model_to_graph(model):
    converter = NNIIRConverter(model)
    return converter.convert()

def torch_model_to_graph(model, input_shape=(1, 3, 224, 224)):
    torch = try_import_torch()
    args = torch.randn(*input_shape)
    if next(model.parameters()).is_cuda:
        args = args.to("cuda")
    converter = TorchConverter(model, args)
    return converter.convert()
    


