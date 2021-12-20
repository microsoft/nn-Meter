# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import random
from .utils import *


def sampling_conv(count):
    ''' 
    Sampling functions for conv kernels based on conv_zoo, which contains configuration values from existing model zoo for conv kernel. 
    The values are stored in prior_zoo/modelzoo_conv.csv.
    '''
    hws, cins, couts, kernel_sizes, _, strides = read_conv_zoo()
    new_cins = sample_based_on_distribution(cins, count)     
    new_couts = sample_based_on_distribution(couts, count)

    # 70% of sampled data are from prior distribution
    count1 = int(count * 0.7)
    new_hws = sample_based_on_distribution(hws, count1)
    new_kernel_size = sample_based_on_distribution(kernel_sizes, count1)
    new_strides = sample_based_on_distribution(strides, count1)
    
    new_kernel_size = data_validation(new_kernel_size, [1, 3, 5, 7])
    new_strides = data_validation(new_strides, [1, 2, 4])
    new_hws = data_validation(new_hws, [1, 3, 7, 8, 13, 14, 27, 28, 32, 56, 112, 224])
    
    # since conv is the largest and most-challenging kernel, we add some frequently used configuration values
    new_hws.extend([112] * int((count - count1) * 0.2) + [56] * int((count - count1) * 0.4) + [28] * int((count - count1) * 0.4)) # frequent settings
    new_kernel_size.extend([5] * int((count - count1) * 0.4) + [7] * int((count - count1) * 0.6)) # frequent settings
    new_strides.extend([2] * int((count - count1) * 0.4) + [1] * int((count - count1) * 0.6)) # frequent settings
    random.shuffle(new_hws)
    random.shuffle(new_strides)
    random.shuffle(new_kernel_size)
    return new_cins, new_couts, new_hws, new_kernel_size, new_strides


def sampling_conv_random(count):
    '''
    '''
    hws = [1, 7, 8, 13, 14, 27, 28, 32, 56, 112, 224]
    kernel_sizes = [1, 3, 5, 7]
    strides = [1, 2, 4]
    
    cins = list(range(3, 2160))
    couts = list(range(16, 2048))
    new_hws = random.sample(hws * int(count / len(hws)) * 10, count)
    new_kernel_sizes = random.sample(kernel_sizes * int(count / len(kernel_sizes) * 10), count)
    new_strides = random.sample(strides * int(count / len(strides) * 10), count)
    new_cins = random.sample(cins * 10, count)
    new_couts = random.sample(couts * 18, count)
    random.shuffle(new_cins)
    random.shuffle(new_couts)
    return new_cins, new_couts, new_hws, new_kernel_sizes, new_strides


def sampling_dwconv(count):
    ''' 
    Sampling functions for dwconv kernels based on dwconv zoo, which contains configuration values from existing model zoo for dwconv kernel. 
    The values are stored in prior_zoo/modelzoo_dwconv.csv.
    '''
    hws, cins, ks, strides = read_dwconv_zoo()
    new_cins = sample_based_on_distribution(cins, count)
   
    count1 = int(count * 0.8)
    new_hws = sample_based_on_distribution(hws,count1)
    new_kernel_sizes = sample_based_on_distribution(ks,count1)
    new_strides = sample_based_on_distribution(strides,count1)
    
    new_hws = data_validation(new_hws, [1, 3, 7, 14, 28, 56, 112, 224])
    new_kernel_sizes = data_validation(new_kernel_sizes, [1, 3, 5, 7])
    new_strides = data_validation(new_strides, [1, 2])
    
    new_hws.extend([112] * int((count - count1) * 0.4) + [56] * int((count - count1) * 0.4) + [28] * int((count - count1) * 0.2))  
    new_kernel_sizes.extend([5] * int((count - count1) * 0.4) + [7] * int((count - count1) * 0.6))
    new_strides.extend([2] * int((count - count1) * 0.5) + [1] * int((count - count1) * 0.5))
    random.shuffle(new_hws)
    random.shuffle(new_kernel_sizes)
    random.shuffle(new_strides)
    return new_cins, new_hws, new_kernel_sizes, new_strides


def sampling_addrelu(count, ratio = 1.8):
    ''' sampling functions for relu
    '''
    _, cins, _, _, _, _, = read_conv_zoo()    
    new_cins = sample_based_on_distribution(cins, count)
    new_cins = [int(x * ratio) for x in new_cins]
    new_hws = [14] * int(count * 0.4) + [7] * int(count * 0.4) + [28] * int(count * 0.2)
    random.shuffle(new_hws)
    return new_hws, new_cins


def sampling_fc(count, fix_cout = 1000):
    '''
    Sampling functions for fc kernels based on fc zoo, which contains configuration values from existing model zoo for fc kernel. 
    The values are stored in prior_zoo/modelzoo_fcs.csv.
    '''
    cins, couts = read_fc_zoo()
    new_cins = sample_based_on_distribution(cins, count)
    if not fix_cout:
        new_couts = sample_based_on_distribution(couts, count)
    else:
        new_couts = [fix_cout] * count
    return new_cins, new_couts


def sampling_pooling(count):
    '''
    Sampling functions for pooling kernels based on pooling zoo, which contains configuration values from existing model zoo for pooling kernel. 
    The values are stored in prior_zoo/modelzoo_pooling.csv.
    '''
    hws, cins, _, _ = read_pool_zoo()
    new_cins = sample_based_on_distribution(cins, count)
    new_hws = sample_based_on_distribution(hws, count)
    new_hws = data_validation(new_hws, [14, 28, 56, 112, 224])
    new_kernel_sizes = [3] * count
    new_strides = [2] * count
    return new_hws, new_cins, new_kernel_sizes, new_strides


def sampling_concats(count):
    ''' sampling functions for concat
    '''
    hws, cins, _, _, _, _ = read_conv_zoo()
    new_hws = sample_based_on_distribution(hws, count)
    new_cins1 = sample_based_on_distribution(cins, count)
    new_cins2 = sample_based_on_distribution(cins, count)
    new_cins3 = sample_based_on_distribution(cins, count)
    new_cins4 = sample_based_on_distribution(cins, count)

    new_hws = data_validation(new_hws, [7, 14, 28, 56])  # current normals
    new_ns = [2] * (count - int(count * 0.4) - int(count * 0.2)) + [3] * int(count * 0.2) + [4] * int(count * 0.4)
    random.shuffle(new_ns)
    return new_hws, new_ns, new_cins1, new_cins2, new_cins3, new_cins4
