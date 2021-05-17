import onnx
from onnx_converter import OnnxConverter
from frozenpb_converter import FrozenPbConverter


def model_to_grapher(model, model_type=None):
    if model_type is None:
        if isinstance(model, onnx.ModelProto):
            model_type = 'onnx'
        else:
            raise ValueError(f'Invalid model: {type(model)}')

    if model_type == 'onnx':
        converter = OnnxConverter(model)
        result = converter.convert()
    elif model_type == 'pb':
        raise NotImplementedError
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

    return result


def model_file_to_grapher(filename, model_type=None):
    if model_type is None:
        if filename.endswith('.onnx'):
            model_type = 'onnx'
        elif filename.endswith('.pb'):
            converter = FrozenPbConverter(filename)
            return converter.get_flatten_grapher()
        else:
            raise ValueError(f'Unknown file type: {filename}')

    if model_type == 'onnx':
        model = onnx.load(filename)
    elif model_type == 'pb':
        raise NotImplementedError
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

    return model_to_grapher(model, model_type)
