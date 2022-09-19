# # hw_candidate = [160, 176, 192, 208, 224]

# # lis = []
# # for hw in hw_candidate:
# #     thw = hw

# #     lis.append(int(thw))
# #     while thw % 2 == 0:
# #         thw = int(thw/2)
# #         lis.append(int(thw))


# import numpy as np
# # lis = list(np.unique(lis))
# # print(list(lis))
# # print(len(lis))

# # import numpy as np
# # lis = [3, 3, 3, 2, 3, 5, 2, 2, 4, 3, 3, 2, 3, 2, 4, 5, 5, 3, 2, 2, 2, 5, 6, 5, 3, 3, 5, 5, 4, 20, 2, 3, 2, 3, 2, 3, 5, 5, 4, 5, 3, 4, 4, 3, 5, 5, 5, 6, 3, 15, 15, 5, 3, 5, 3, 5, 3, 2, 3, 4, 5, 4, 4, 4, 7, 3]
# # print(len(lis))
# # print(f'{round(np.mean(lis), 2)} +- {round(np.std(lis), 2)} ({np.min(lis)}-{np.max(lis)})')
# # print(f'{round(np.quantile(lis, 0.5), 2)} ({round(np.quantile(lis, 0.75), 2)}, {round(np.quantile(lis, 0.25), 2)}, {round(np.quantile(lis, 0.75) - np.quantile(lis, 0.25), 2)})')


# # data = [[25, 40], [44, 22]] # 1
# # data = [[42, 17], [40, 26]] # 2
# # data = [[22, 6], [31, 3]] # 3
# # data = [[15, 44], [16, 50]]
# data = [[31, 3], [2, 1]] # 4
# data = [[22, 6], [31, 3]]
# data = [[22, 6], [2, 1]]

# # data = [
# #     [[15, 46], [16, 50]],
# #     [[10, 51], [28, 38]],
# #     [[33, 28], [16, 50]],
# #     [[0, 61], [2, 64]],
# #     [[3, 58], [4, 61]]
# # ]
# # data = [
# #     [[15, 44], [16, 50]],
# #     [[27, 32], [24, 42]],
# #     [[14, 45], [22, 44]],
# #     [[3, 56], [4, 62]]
# # ]

# import scipy.stats as stats
# print(stats.fisher_exact(data))
# # for da in data:
# #     print(stats.fisher_exact(da))


# # from nn_meter.predictor.quantize_block_predictor import BlockLatencyPredictor
# # a = BlockLatencyPredictor("tflite27_cpu_int8")
# # print(a.get_latency("MobileNetV3Block", 5, 320, 1280, 1, 3, 1, "hswish"))
# # import pdb; pdb.set_trace()



import sys
import random
import signal
import time




def do_stuff(n):
    time.sleep(n)


def main():
    # signal.signal(signal.SIGINT, handle_sigint)
    

    max_duration = 5
    
    for duration in [2, 10]:
        try:
            print('duration = {}: '.format(duration), end='', flush=True)
            signal.alarm(max_duration + 1)
            do_stuff(duration)
            signal.alarm(0)
        except TimeoutError as exc:
            print('{}: {}'.format(exc.__class__.__name__, exc))
        else:
            print('slept for {}s'.format(duration))


if __name__ == '__main__':
    print("now")
    a = time.time()
    main()
    print(time.time()-a)
    print("now")