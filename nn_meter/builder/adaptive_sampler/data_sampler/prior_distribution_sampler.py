# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import random
import numpy as np

from utils import read_conv_zoo, read_dwconv_zoo, read_fc_zoo, read_pool_zoo, sample_based_on_distribution, data_clean


def sampling_conv(count):
    ''' 
    Sampling functions for conv kernels based on conv_zoo, which contains configuration values from existing model zoo for conv kernel. 
    The values are stored in prior_zoo/modelzoo_conv.csv.
    '''
    hws, cins, couts, ks, groups, strides = read_conv_zoo()
    newcins = sample_based_on_distribution(cins, count)     
    newcouts = sample_based_on_distribution(couts, count)

    # 70% of sampled data are totally from prior distribution
    count1 = int(count * 0.7)
    newhws = sample_based_on_distribution(hws, count1)
    newks = sample_based_on_distribution(ks, count1)
    newstrides = sample_based_on_distribution(strides, count1)
    
    newks = data_clean(newks, [1, 3, 5, 7])
    newstrides = data_clean(newstrides, [1, 2, 4])
    newhws = data_clean(newhws, [1, 3, 7, 8, 13, 14, 27, 28, 32, 56, 112, 224])
    
    # since conv is the largest and most-challenging kernel, we add some frequently used configuration values
    newhws.extend([112] * int((count - count1) * 0.2) + [56] * int((count - count1) * 0.4) + [28] * int((count - count1) * 0.4)) # frequent settings
    newks.extend([5] * int((count-count1) * 0.4) + [7] * int((count-count1) * 0.6)) # frequent settings
    newstrides.extend([2] * int((count - count1) * 0.4) + [1] * int((count - count1) * 0.6)) # frequent settings
    random.shuffle(newhws)
    random.shuffle(newstrides)
    random.shuffle(newks)
    # print(len(newcins),len(newstrides),len(newks),len(newhws)) #TODO: change to log
    return newcins, newcouts, newhws, newks, newstrides


def sampling_conv_random(count):
    '''
    '''
    hws = [224, 112, 56, 32, 28, 27, 14, 13, 8, 7, 1]
    ks = [1, 3, 5, 7]
    s = [1, 2, 4]
    
    cins = list(range(3,2160))
    couts = list(range(16,2048))
    newhws = random.sample(hws * int(count / len(hws)) * 10, count)
    newks = random.sample(ks * int(count / len(ks) * 10), count)
    newstrides = random.sample(s * int(count / len(s) * 10), count)
    newcins = random.sample(cins * 10, count)
    newcouts = random.sample(couts * 18, count)
    random.shuffle(newcins)
    random.shuffle(newcouts)
    keys = []
    for index in range(len(newhws)):
        key = '_'.join(str(x) for x in[newhws[index],newks[index],newstrides[index],newcins[index],newcouts[index]])
        if key not in keys:
            keys.append(key)
    # print(len(keys)) #TODO: change to log
    return newcins, newcouts, newhws, newks, newstrides


def sampling_dwconv(count):
    ''' 
    Sampling functions for dwconv kernels based on dwconv zoo, which contains configuration values from existing model zoo for dwconv kernel. 
    The values are stored in prior_zoo/modelzoo_dwconv.csv.
    '''
    hws,cs,ks,strides = read_dwconv_zoo()
    newcs = sample_based_on_distribution(cs,count)
   
    count1 = int(count*0.8)
    newhws = sample_based_on_distribution(hws,count1)
    newks = sample_based_on_distribution(ks,count1)
    newstrides = sample_based_on_distribution(strides,count1)
    
    newhws = data_clean(newhws,[1,3,7,14,28,56,112,224])
    newks = data_clean(newks,[1,3,5,7])
    newstrides = data_clean(newstrides,[1,2])
    
    newhws.extend([112]*int((count-count1)*0.4)+[56]*int((count-count1)*0.4)+[28]*int((count-count1)*0.2))  
    newks.extend([5]*int((count-count1)*0.4)+[7]*int((count-count1)*0.6))
    newstrides.extend([2]*int((count-count1)*0.5)+[1]*int((count-count1)*0.5))
    random.shuffle(newhws)
    random.shuffle(newks)
    random.shuffle(newstrides)
    # print(len(newcs),len(newhws),len(newks),len(newstrides)) #TODO: change to log
   
    return newcs,newhws,newks,newstrides

def sampling_addrelu(count, ratio = 1.8):
    ''' sampling functions for relu
    '''
    hws, cins, couts, ks, groups, strides = read_conv_zoo()    
    newcs = sample_based_on_distribution(cins, count)
    newcs = [int(x * ratio) for x in newcs]
    newhws = [14] * int(count * 0.4) + [7] * int(count * 0.4) + [28] * int(count * 0.2)
    random.shuffle(newhws)
    # print(newcs) #TODO: change to log
    return newhws, newcs, newcs


def sampling_fc(count, fix_cout = 1000):
    '''
    Sampling functions for fc kernels based on fc zoo, which contains configuration values from existing model zoo for fc kernel. 
    The values are stored in prior_zoo/modelzoo_fcs.csv.
    '''
    cins,couts = read_fc_zoo()
    newcins = sample_based_on_distribution(cins, count)
    if not fix_cout:
        newcouts = sample_based_on_distribution(couts, count)
    else:
        newcouts = [fix_cout] * count
    return newcins, newcouts


def sampling_pooling(count):
    '''
    Sampling functions for pooling kernels based on pooling zoo, which contains configuration values from existing model zoo for pooling kernel. 
    The values are stored in prior_zoo/modelzoo_pooling.csv.
    '''
    hws, cins, ks, strides = read_pool_zoo()
    newcins = sample_based_on_distribution(cins, count)
    newhws = sample_based_on_distribution(hws, count)
    newks = sample_based_on_distribution(ks, count)
    newstrides = sample_based_on_distribution(strides, count)

    newhws = data_clean(newhws, [14, 28, 56, 112, 224])
    newks = data_clean(newks, [3])
    newstrides = data_clean(newstrides, [2])
    return newhws, newcins, newks, newstrides


def sampling_concats(count):
    ''' sampling functions for concat
    '''
    hws, cins, couts, ks, groups, strides = read_conv_zoo()
    newcins1 = sample_based_on_distribution(cins, count)
    newcins2 = sample_based_on_distribution(cins, count)
    newcins3 = sample_based_on_distribution(cins, count)
    newcins4 = sample_based_on_distribution(cins, count)

    newhws = sample_based_on_distribution(hws, count)
    newhws = data_clean(newhws, [7, 14, 28, 56])  # current normals
    newns = [2] * int(count * 0.4) + [3] * int(count * 0.2) + [4] * int(count * 0.4)
    if len(newns) < count:
        da = [2] * (count - len(newns))
        newns.extend(da)
    random.shuffle(newns)
    return newhws, newns, newcins1, newcins2, newcins3, newcins4
    




