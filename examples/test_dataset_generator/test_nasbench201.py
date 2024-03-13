config = "|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_1x1~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|"
output = "/data1/jiahang/working/pixel4_int8_workspace/code/test.h5"
from nn_meter.dataset.generator.networks.tf_network.nasbench201 import nasbench201
nasbench201(config, output)




from nas_201_api import NASBench201API as API
api = API("/data/jiahang/working/NAS-Bench-201/NAS-Bench-201-v1_0-e61699.pth")
index = api.query_index_by_arch(config)
print(index)
api.show(index)