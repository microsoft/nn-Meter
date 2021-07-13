# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import onnx
import torch
import json
from .onnx_converter import OnnxConverter
from .frozenpb_converter import FrozenPbConverter
from .torch_converter import TorchConverter
from .torch_converter.converter import NNIIRConverter


def model_to_graph(model, model_type, input_shape=(1, 3, 224, 224)):
    """
    @params:

    input_shape: only accessed when model_type == 'torch'
    """
    if model_type == "onnx":
        converter = OnnxConverter(model)
        result = converter.convert()
    elif model_type == "pb":
        raise NotImplementedError
    elif model_type == "torch":
        args = torch.randn(*input_shape)
        if next(model.parameters()).is_cuda:
            args = args.to("cuda")
        converter = TorchConverter(model, args)
        result = converter.convert()
    elif model_type == "nni":
        converter = NNIIRConverter(model)
        result = converter.convert()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return result


def model_file_to_graph(filename, model_type=None, input_shape=(1, 3, 224, 224)):
    """
    @params:

    input_shape: only accessed when model_type == 'torch'
    """
    if model_type is None:
        if filename.endswith(".onnx"):
            model_type = "onnx"
        elif filename.endswith(".pb"):
            model_type = "pb"
        elif filename.endswith(".json"):
            model_type = "json"
        elif filename.endswith(".pth") or filename.endswith(".pt"):
            model_type = "torch"
        else:
            raise ValueError(f"Unknown file type: {filename}")

    if model_type == "onnx":
        model = onnx.load(filename)
        return model_to_graph(model, model_type)
    elif model_type == "pb":
        converter = FrozenPbConverter(filename)
        return converter.get_flatten_graphe()
    elif model_type == "json":
        with open(filename, "r") as fp:
            return json.load(fp)
    elif model_type == "torch":
        model = torch.load(filename)
        return model_to_graph(model, model_type, input_shape)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
