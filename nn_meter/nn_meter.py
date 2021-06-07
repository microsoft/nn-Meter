# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from .prediction.predictors.predict_by_kernel import nn_predict
from .prediction.predictors.predict_by_kernel import main_kernel_predict
from .kerneldetection import KernelDetector
from .ir_converters import model_to_graph, model_file_to_graph
from .prediction.load_predictors import load_lat_predictors


def load_latency_predictors(config):
    kernel_predictors, fusionrule = load_lat_predictors(config['predictor'])

    nnmeter = nnMeter(kernel_predictors,fusionrule)
    return nnmeter


class nnMeter:
    def __init__(self, predictors, fusionrule):
        self.kernel_predictors = predictors
        self.fusionrule = fusionrule
        self.kd = KernelDetector(self.fusionrule)

    def predict(self, model, model_type=None):
        '''
        @params:

        model: a pytorch/onnx/tensorflow model object or a str containing path to the model file
        '''
        if isinstance(model, str):
            graph = model_file_to_graph(model, model_type)
        else:
            graph = model_to_graph(model, model_type)
        self.kd.load_graph(graph)
        #mid=self.__getmodelname__(model)
        mid = "test"
        kernel_result = {mid: self.kd.kernels}
        py = nn_predict(self.kernel_predictors, kernel_result)
        return py
