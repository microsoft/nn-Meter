# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import logging
from packaging import version
from .utils import load_config_file, loading_to_local, loading_customized_predictor
from .prediction.predict_by_kernel import nn_predict
from nn_meter.kernel_detector import KernelDetector
from nn_meter.utils import get_user_data_folder
from nn_meter.ir_converter import model_file_to_graph, model_to_graph
logging = logging.getLogger("nn-Meter")


__predictors_cfg_filename__ = 'predictors.yaml'


def list_latency_predictors():
    """return the list of latency predictors specified in nn_meter/configs/predictors.yaml
    """
    return load_config_file(__predictors_cfg_filename__)


def load_predictor_config(predictor_name: str, predictor_version: float = None):
    """
    return the information of the predictor model according to the given predictor name and version
    @params:

    predictor_name: string to specify the name of the target latency predictor. All built-in predictors can be viewed by nn_meter.list_latency_predictors() 
        or through the config file in nn_meter/configs/predictors.yaml.
    
    predictor_version: string to specify the version of the target latency predictor. If not specified (default as None), the lateast version of the 
        predictor will be loaded.
    """
    config = load_config_file(__predictors_cfg_filename__)
    preds_info = [p for p in config if p['name'] == predictor_name and (predictor_version is None or p['version'] == predictor_version)]
    n_preds = len(preds_info)
    if n_preds == 1:
        return preds_info[0]
    elif n_preds > 1:
        # find the latest version of the predictor
        latest_version, latest_version_idx = version.parse(str(preds_info[0]['version'])), 0
        for i in range(1, n_preds):
            if version.parse(str(preds_info[i]['version'])) > latest_version:
                latest_version = version.parse(str(preds_info[i]['version']))
                latest_version_idx = i
        print(f'WARNING: There are multiple version for {predictor_name}, use the latest one ({str(latest_version)})')
        return preds_info[latest_version_idx]
    else:
        raise NotImplementedError('No predictor that meets the required name and version, please try again.')


def load_latency_predictor(predictor_name: str, predictor_version: float = None):
    """ 
    return the predictor model according to the given predictor name and version
    @params:

    predictor_name: string to specify the name of the target latency predictor. All built-in predictors can be viewed by nn_meter.list_latency_predictors() 
        or through the config file in ~/.nn_meter/config/predictors.yaml.
    
    predictor_version: string to specify the version of the target latency predictor. If not specified (default as None), the lateast version of the 
        predictor will be loaded.
    """
    user_data_folder = get_user_data_folder()
    pred_info = load_predictor_config(predictor_name, predictor_version)
    if "download" in pred_info:
        kernel_predictors, fusionrule = loading_to_local(pred_info, os.path.join(user_data_folder, 'predictor'))
    else:
        kernel_predictors, fusionrule = loading_customized_predictor(pred_info)
        
    return nnMeterPredictor(kernel_predictors, fusionrule)


class nnMeterPredictor:
    def __init__(self, predictors, fusionrule):
        self.kernel_predictors = predictors
        self.fusionrule = fusionrule
        self.kd = KernelDetector(self.fusionrule)

    def predict(
        self, model, model_type, input_shape=(1, 3, 224, 224), apply_nni=False
    ):
        """
        return the predicted latency in microseconds (ms)
        @params:

        model: the model to be predicted, allowed file include
            - the path to a saved tensorflow model file (*.pb), `model_type` must be set to "pb"
            - pytorch model object (nn.Module), `model_type` must be set to "torch"
            - ONNX model object or the path to a saved ONNX model file (*.onnx), `model_type` must be set to "onnx"
            - dictionary object following nn-Meter-IR format, `model_type` must be set to "nnmeter-ir"
            - dictionary object following NNI-IR format, `model_type` must be set to "nni-ir"
            
        model_type: string to specify the type of parameter model, allowed items are ["pb", "torch", "onnx", "nnmeter-ir", "nni-ir"]
      
        input_shape: the shape of input tensor for inference (if necessary), a random tensor according to the shape will be generated and used. This parameter is only 
        accessed when model_type == 'torch'

        apply_nni: switch the torch converter used for torch model parsing. If apply_nni==True, NNI-based converter is used for torch model conversion, which requires 
            nni>=2.4 installation and should use nn interface from NNI `import nni.retiarii.nn.pytorch as nn` to define the PyTorch modules. Otherwise Onnx-based torch 
            converter is used, which requires onnx installation (well tested version is onnx==1.9.0). NNI-based converter is much faster while the conversion is unstable 
            as it could fail in some case. Onnx-based converter is much slower but stable compared to NNI-based converter. This parameter is only accessed when 
            model_type == 'torch'
        """
        logging.info("Start latency prediction ...")
        if isinstance(model, str):
            graph = model_file_to_graph(model, model_type, input_shape, apply_nni=apply_nni)
        else:
            graph = model_to_graph(model, model_type, input_shape=input_shape, apply_nni=apply_nni)
        
        # logging.info(graph)
        self.kd.load_graph(graph)

        py = nn_predict(self.kernel_predictors, self.kd.get_kernels()) # in unit of ms
        logging.info(f"Predict latency: {py} ms")
        return py
