import os
from glob import glob
import nn_meter

predictor_name = 'adreno640gpu_tflite21' # user can change text here to test other predictors
predictor_version = 1.0

# # download data and unzip
# ppath = "/data/jiahang/working/data/nnmeter_ir_graphs"

# test_model_list = glob(ppath + "/**.json")

# # load predictor
# predictor = nn_meter.load_latency_predictor(predictor_name, predictor_version)

# # predict latency
# result = {}
# for test_model in test_model_list:
#     latency = predictor.detailed_predict(test_model, model_type="nnmeter-ir") # in unit of ms
#     result[os.path.basename(test_model)] = latency
#     print(f'[RESULT] predict latency for {os.path.basename(test_model)}: {latency} ms')



import os
from glob import glob
import nn_meter

# download data and unzip
ppath = os.path.join("/data/jiahang/working/data/pb_models")
# url = "https://github.com/microsoft/nn-Meter/releases/download/v1.0-data/pb_models.zip"
# nn_meter.download_from_url(url, ppath)

test_model_list = glob(ppath + "/**.pb")

# load predictor
predictor = nn_meter.load_latency_predictor(predictor_name, predictor_version)

# predict latency
result = {}
for test_model in test_model_list:
    print(test_model)
    latency, block_res = predictor.detailed_predict(test_model, model_type="pb") # in unit of ms
    result[os.path.basename(test_model)] = latency
    print(f'[RESULT] predict latency for {test_model}: {latency} ms')
    for item in block_res:
        print(item)
    print()
    break




# import os
# from glob import glob
# import nn_meter

# # download data and unzip
# ppath = os.path.join("/data/jiahang/working/data/onnx_models")
# if not os.path.isdir(ppath):
#     os.mkdir(ppath)
#     url = "https://github.com/microsoft/nn-Meter/releases/download/v1.0-data/onnx_models.zip"
#     nn_meter.download_from_url(url, ppath)

# test_model_list = glob(ppath + "/**.onnx")

# # load predictor
# predictor = nn_meter.load_latency_predictor(predictor_name, predictor_version)

# # predict latency
# result = {}
# for test_model in test_model_list:
#     print(test_model)
#     latency, block_res = predictor.detailed_predict(test_model, model_type="onnx") # in unit of ms
#     result[os.path.basename(test_model)] = latency
#     print(f'[RESULT] predict latency for {os.path.basename(test_model)}: {latency} ms')
#     for item in block_res:
#         print(item)
#     print()
#     # break


# import os, sys
# sys.path.append("/data/jiahang/working/nn-Meter/tests/integration_test/data")
# import torchmodels as models 
# import nn_meter

# torchvision_models = {
#     "resnet18": models.resnet18(),
#     "alexnet": models.alexnet(),
#     "vgg16": models.vgg16(),
#     "squeezenet": models.squeezenet1_0(),
#     "densenet161": models.densenet161(),
#     "inception_v3": models.inception_v3(),
#     "googlenet": models.googlenet(),
#     "shufflenet_v2": models.shufflenet_v2_x1_0(),
#     "mobilenet_v2": models.mobilenet_v2(),
#     "resnext50_32x4d": models.resnext50_32x4d(),
#     "wide_resnet50_2": models.wide_resnet50_2(),
#     "mnasnet": models.mnasnet1_0()
# }


# for model_name in torchvision_models:
#     latency = predictor.detailed_predict(torchvision_models[model_name], model_type="torch", input_shape=(1, 3, 224, 224)) 
#     print(f'[RESULT] predict latency for {model_name}: {latency} ms')
#     print()
#     print()
#     # break
