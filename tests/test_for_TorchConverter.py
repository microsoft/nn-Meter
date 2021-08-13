import models
from nn_meter import load_latency_predictor
# from nn_meter.ir_converters.utils import torch_model_to_graph

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
    # for model_name in torchvision_zoo_dict:
    #     print("################################# ", model_name, " #################################")
    #     model = eval(torchvision_zoo_dict[model_name])
    #     try:
    #         predictors.predict(model, "torch")
    #     except:
    #         pass
    
    model_name = "shufflenet_v2"
    print("################################# ", model_name, " #################################")
    model = eval(torchvision_zoo_dict[model_name])
    predictors.predict(model, "torch")
    