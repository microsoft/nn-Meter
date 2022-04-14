hw_candidate = [160, 176, 192, 208, 224]

lis = []
for hw in hw_candidate:
    thw = hw

    lis.append(int(thw))
    while thw % 2 == 0:
        thw = int(thw/2)
        lis.append(int(thw))


import numpy as np
lis = list(np.unique(lis))
print(list(lis))
print(len(lis))