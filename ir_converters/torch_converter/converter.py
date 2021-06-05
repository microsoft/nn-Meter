import torch
import onnx
import tempfile
from ir_converters.onnx_converter import OnnxConverter


class TorchConverter(OnnxConverter):
    def __init__(self, model, args):
        '''
        @params
        args: model input, refer to https://pytorch.org/docs/stable/onnx.html#example-end-to-end-alexnet-from-pytorch-to-onnx for more information.
        '''
        with tempfile.TemporaryFile() as fp:
            torch.onnx.export(model, args, fp)
            fp.seek(0)
            model = onnx.load(fp, load_external_data=False)

        super().__init__(model)
