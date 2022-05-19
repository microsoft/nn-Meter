# hw_candidate = [160, 176, 192, 208, 224]

# lis = []
# for hw in hw_candidate:
#     thw = hw

#     lis.append(int(thw))
#     while thw % 2 == 0:
#         thw = int(thw/2)
#         lis.append(int(thw))


# import numpy as np
# lis = list(np.unique(lis))
# print(list(lis))
# print(len(lis))

import numpy as np
lis = [3, 3, 3, 2, 3, 5, 2, 2, 4, 3, 3, 2, 3, 2, 4, 5, 5, 3, 2, 2, 2, 5, 6, 5, 3, 3, 5, 5, 4, 20, 2, 3, 2, 3, 2, 3, 5, 5, 4, 5, 3, 4, 4, 3, 5, 5, 5, 6, 3, 15, 15, 5, 3, 5, 3, 5, 3, 2, 3, 4, 5, 4, 4, 4, 7, 3]
print(len(lis))
print(f'{round(np.mean(lis), 2)} +- {round(np.std(lis), 2)} ({np.min(lis)}-{np.max(lis)})')
print(f'{round(np.quantile(lis, 0.5), 2)} ({round(np.quantile(lis, 0.75), 2)}, {round(np.quantile(lis, 0.25), 2)}, {round(np.quantile(lis, 0.75) - np.quantile(lis, 0.25), 2)})')


# data = [[25, 40], [44, 22]] # 1
data = [[42, 17], [40, 26]] # 2
# data = [[22, 6], [31, 3]] # 3
# data = [[15, 44], [16, 50]]
# data = [[31, 3], [2, 1]] # 4

import scipy.stats as stats
print(stats.fisher_exact(data))


from nn_meter.predictor.quantize_block_predictor import BlockLatencyPredictor
a = BlockLatencyPredictor("tflite27_cpu_int8")
print(a.get_latency("MobileNetV3Block", 5, 320, 1280, 1, 3, 1, "hswish"))
import pdb; pdb.set_trace()
