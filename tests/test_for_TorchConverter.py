import models
from nn_meter import load_latency_predictor, model_to_graph
import json
import torch
from nni.retiarii.converter import convert_to_graph
from nni.retiarii.converter.graph_gen import GraphConverter, GraphConverterWithShape
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (bytes, bytearray)):
            return obj.decode("utf-8")
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    base_predictor = 'cortexA76cpu_tflite21'
    # predictors = load_latency_predictor(base_predictor)

    torchvision_zoo_dict = {
        'resnet18': 'models.resnet18()',
        'alexnet': 'models.alexnet()',
        'vgg16': 'models.vgg16()',
        'squeezenet': 'models.squeezenet1_0()',
        'densenet161': 'models.densenet161()',
        'inception_v3': 'models.inception_v3()',
        'googlenet': 'models.googlenet()',
        'shufflenet_v2': 'models.shufflenet_v2_x1_0()',
        'mobilenet_v2': 'models.mobilenet_v2()',  # noqa: F841
        'resnext50_32x4d': 'models.resnext50_32x4d()',
        'wide_resnet50_2': 'models.wide_resnet50_2()',
        'mnasnet': 'models.mnasnet1_0()',
    }
    model = models.shufflenet_v2_x1_0()
    input_shape=(1, 3, 224, 224)
    example_inputs = torch.randn(*input_shape)

    model_name = 'mnasnet'
    # for model_name in torchvision_zoo_dict:
    print("################################# ", model_name, " #################################")
    model = eval(torchvision_zoo_dict[model_name])
    # predictors.predict(model, "torch")
    
    print(torch.jit.script(model))
    graph = model_to_graph(model, "torch")
    # if not str.endswith(filename, '.json'): filename += '.json'
    with open(f"{model_name}.json", "w+") as fp:
        json.dump(graph,
            fp,
            indent=4,
            skipkeys=True,
            sort_keys=True,
            cls=NumpyEncoder,
        )

    
    # model_name = "shufflenet_v2"
    # print("################################# ", model_name, " #################################")
    # model = eval(torchvision_zoo_dict[model_name])
    # predictors.predict(model, "torch")
    