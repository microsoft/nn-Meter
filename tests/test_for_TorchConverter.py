import torchmodels as models
from nn_meter import load_latency_predictor, model_to_graph
import json
import torch

if __name__ == "__main__":
    base_predictor = 'cortexA76cpu_tflite21'
    predictors = load_latency_predictor(base_predictor)

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

    model_name = 'shufflenet_v2'
    
    print("################################# ", model_name, " #################################")
    model = eval(torchvision_zoo_dict[model_name])
    predictors.predict(model, "torch")
