# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from .prediction.predictors.predict_by_kernel import nn_predict
from .kerneldetection import KernelDetector
from .ir_converters import model_to_graph, model_file_to_graph
from .prediction.load_predictors import loading_to_local


def load_latency_predictors(config):
    kernel_predictors, fusionrule = loading_to_local(config)
    nnmeter = nnMeter(kernel_predictors,fusionrule)
    return nnmeter


class nnMeter:
    def __init__(self, predictors, fusionrule):
        self.kernel_predictors = predictors
        self.fusionrule = fusionrule
        self.kd = KernelDetector(self.fusionrule)

    def predict(self, model, model_type=None,input_shape=(1, 3, 224, 224),modelname='test'):
        '''
        @params:

        model: a pytorch/onnx/tensorflow model object or a str containing path to the model file
        '''
        if isinstance(model, str):
            graph = model_file_to_graph(model, model_type)
            modelname=model.split('/')[-1].split('.')[0]
        else:
            graph = model_to_graph(model, model_type,input_shape=input_shape)
        self.kd.load_graph(graph)
        
        
        kernel_result = {modelname: self.kd.kernels}
        py = nn_predict(self.kernel_predictors, kernel_result)
        return py
