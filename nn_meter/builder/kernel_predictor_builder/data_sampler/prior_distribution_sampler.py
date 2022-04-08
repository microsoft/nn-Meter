# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import random
import numpy as np
from .prior_config_lib.utils import *


def inverse_transform_sampling(data, n_bins = 40, n_samples = 1000):
    ''' calculate inversed cdf, for sampling by possibility
    '''
    import scipy.interpolate as interpolate
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    r = np.random.rand(n_samples)
    data = inv_cdf(r)
    ndata = [int(x) for x in data]
    return ndata


def sample_based_on_distribution(data, count):
    ''' use data to calculate a inversed cdf, and sample `count` data from such distribution
    '''
    return inverse_transform_sampling(data, n_samples=count)


def data_validation(data, cdata):
    ''' convert sampled data to valid configuration, e.g.,: kernel size = 1, 3, 5, 7

    @params:
    data: the origin data value.
    cdata: valid configuration value.
    '''
    newlist = []
    for da in cdata:
        value = [abs(da - x) for x in data]
        newlist.append(value)

    newlist = list(np.asarray(newlist).T)    
    cda = [list(d).index(min(d)) for d in newlist]
    redata = [cdata[x] for x in cda]
    return redata


def sampling_conv(count):
    ''' 
    Sampling configs for conv kernels based on conv_zoo, which contains configuration values from existing model zoo for conv kernel. 
    The values are stored in prior_config_lib/conv.csv.
    Returned params include: (hw, cin, cout, kernel_size, strides)
    '''
    hws, cins, couts, kernel_sizes, strides = read_conv_zoo()
    new_cins = sample_based_on_distribution(cins, count)
    new_couts = sample_based_on_distribution(couts, count)

    # 70% of sampled data are from prior distribution
    count1 = int(count * 0.7)
    new_hws = sample_based_on_distribution(hws, count1)
    new_kernel_sizes = sample_based_on_distribution(kernel_sizes, count1)
    new_strides = sample_based_on_distribution(strides, count1)

    new_kernel_sizes = data_validation(new_kernel_sizes, [1, 3, 5, 7])
    new_strides = data_validation(new_strides, [1, 2, 4])
    new_hws = data_validation(new_hws, [1, 3, 7, 8, 13, 14, 27, 28, 32, 56, 112, 224])
    
    # since conv is the largest and most-challenging kernel, we add some frequently used configuration values
    new_hws.extend([112] * int((count - count1) * 0.2) + [56] * int((count - count1) * 0.4) + [28] * int((count - count1) * 0.4)) # frequent settings
    new_kernel_sizes.extend([5] * int((count - count1) * 0.4) + [7] * int((count - count1) * 0.6)) # frequent settings
    new_strides.extend([2] * int((count - count1) * 0.4) + [1] * int((count - count1) * 0.6)) # frequent settings
    random.shuffle(new_hws)
    random.shuffle(new_strides)
    random.shuffle(new_kernel_sizes)

    ncfgs = []
    for hw, cin, cout, kernel_size, stride in zip(new_hws, new_cins, new_couts, new_kernel_sizes, new_strides):
        c = {
            'HW': hw,
            'CIN': cin,
            'COUT': cout,
            'KERNEL_SIZE': kernel_size,
            'STRIDES': stride,
        }
        ncfgs.append(c)
    return ncfgs


def sampling_conv_random(count):
    ''' sampling configs for conv kernels based on random
    Returned params include: (hw, cin, cout, kernel_size, strides)
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

    ncfgs = []
    for hw, cin, cout, kernel_size, stride in zip(new_hws, new_cins, new_couts, new_kernel_sizes, new_strides):
        c = {
            'HW': hw,
            'CIN': cin,
            'COUT': cout,
            'KERNEL_SIZE': kernel_size,
            'STRIDES': stride,
        }
        ncfgs.append(c)
    return ncfgs


def sampling_dwconv(count):
    ''' 
    Sampling configs for dwconv kernels based on dwconv zoo, which contains configuration values from existing model zoo for dwconv kernel. 
    The values are stored in prior_config_lib/dwconv.csv.
    Returned params include: (hw, cin, kernel_size, strides)
    '''
    hws, cins, ks, strides = read_dwconv_zoo()
    new_cins = sample_based_on_distribution(cins, count)
   
    count1 = int(count * 0.8)
    new_hws = sample_based_on_distribution(hws,count1)
    new_kernel_sizes = sample_based_on_distribution(ks, count1)
    new_strides = sample_based_on_distribution(strides, count1)
    
    new_hws = data_validation(new_hws, [1, 3, 7, 14, 28, 56, 112, 224])
    new_kernel_sizes = data_validation(new_kernel_sizes, [1, 3, 5, 7])
    new_strides = data_validation(new_strides, [1, 2])
    
    new_hws.extend([112] * int((count - count1) * 0.4) + [56] * int((count - count1) * 0.4) + [28] * int((count - count1) * 0.2))  
    new_kernel_sizes.extend([5] * int((count - count1) * 0.4) + [7] * int((count - count1) * 0.6))
    new_strides.extend([2] * int((count - count1) * 0.5) + [1] * int((count - count1) * 0.5))
    random.shuffle(new_hws)
    random.shuffle(new_kernel_sizes)
    random.shuffle(new_strides)

    ncfgs = []
    for hw, cin, kernel_size, stride in zip(new_hws, new_cins, new_kernel_sizes, new_strides):
        c = {
            'HW': hw,
            'CIN': cin,
            'KERNEL_SIZE': kernel_size,
            'STRIDES': stride,
        }
        ncfgs.append(c)
    return ncfgs


def sampling_fc(count, fix_cout = 1000):
    '''
    Sampling configs for fc kernels based on fc zoo, which contains configuration values from existing model zoo for fc kernel. 
    The values are stored in prior_config_lib/fcs.csv.
    Returned params include: (cin, cout)
    '''
    cins, couts = read_fc_zoo()
    new_cins = sample_based_on_distribution(cins, count)
    if not fix_cout:
        new_couts = sample_based_on_distribution(couts, count)
    else:
        new_couts = [fix_cout] * count
    ncfgs = []
    for cin, cout in zip(new_cins, new_couts):
        c = {
            'CIN': cin,
            'COUT': cout,
        }
        ncfgs.append(c)
    return ncfgs


def sampling_pooling(count):
    '''
    Sampling configs for pooling kernels based on pooling zoo, which contains configuration values from existing model zoo for pooling kernel. 
    The values are stored in prior_config_lib/pooling.csv.
    Returned params include: (hw, cin, kernel_size, pool_strides)
    '''
    hws, cins, kernel_size, strides = read_pool_zoo()
    new_cins = sample_based_on_distribution(cins, count)
    new_hws = sample_based_on_distribution(hws, count)
    new_hws = data_validation(new_hws, [14, 28, 56, 112, 224])
    new_kernel_sizes = list(kernel_size) * (count // len(kernel_size) + 1)
    new_kernel_sizes = data_validation(new_kernel_sizes, [2, 3])
    random.shuffle(new_kernel_sizes)
    new_strides = list(strides) * (count // len(strides) + 1)
    new_strides = data_validation(new_strides, [1, 2])
    random.shuffle(new_strides)
    
    ncfgs = []
    for hw, cin, kernel_size, stride in zip(new_hws, new_cins, new_kernel_sizes, new_strides):
        c = {
            'HW': hw,
            'CIN': cin,
            'KERNEL_SIZE': kernel_size,
            'STRIDES': stride,
        }
        ncfgs.append(c)
    return ncfgs


def sampling_hw_cin(count):
    ''' sampling configs for kernels with hw and cin parameter
    Returned params include: (hw, cin)
    '''
    hws, cins, _, _, _ = read_conv_zoo()
    new_cins = sample_based_on_distribution(cins, count)
   
    count1 = int(count * 0.8)
    new_hws = sample_based_on_distribution(hws,count1)
    new_hws = data_validation(new_hws, [1, 3, 7, 14, 28, 56, 112, 224])
    new_hws.extend([112] * int((count - count1) * 0.4) + [56] * int((count - count1) * 0.4) + [28] * int((count - count1) * 0.2))
    random.shuffle(new_hws)

    ncfgs = []
    for hw, cin in zip(new_hws, new_cins):
        c = {
            'HW': hw,
            'CIN': cin,
        }
        ncfgs.append(c)
    return ncfgs


def sampling_hw_cin_even(count):
    ''' sampling configs for kernels with hw and cin (only even values) parameter, in case for split / se / channelshuffle
    Returned params include: (hw, cin)
    '''
    hws, cins, _, _, _ = read_conv_zoo()
    new_cins = sample_based_on_distribution(cins, count)
   
    count1 = int(count * 0.8)
    new_hws = sample_based_on_distribution(hws,count1)
    new_hws = data_validation(new_hws, [1, 3, 7, 14, 28, 56, 112, 224])
    new_hws.extend([112] * int((count - count1) * 0.4) + [56] * int((count - count1) * 0.4) + [28] * int((count - count1) * 0.2))
    random.shuffle(new_hws)

    ncfgs = []
    for hw, cin in zip(new_hws, new_cins):
        c = {
            'HW': hw,
            'CIN': cin + 1 if cin % 2 else cin,
        }
        ncfgs.append(c)
    return ncfgs


def sampling_concats(count):
    ''' sampling functions for concat kernel
    Returned params include: (hw, ns, cin1, cin2, cin3, cin4), ns are in [2, 4]
    '''
    hws, cins, _, _, _ = read_conv_zoo()
    new_hws = sample_based_on_distribution(hws, count)
    new_cins1 = sample_based_on_distribution(cins, count)
    new_cins2 = sample_based_on_distribution(cins, count)
    new_cins3 = sample_based_on_distribution(cins, count)
    new_cins4 = sample_based_on_distribution(cins, count)

    new_hws = data_validation(new_hws, [7, 14, 28, 56])  # current normals
    new_ns = [2] * (count - int(count * 0.4) - int(count * 0.2)) + [3] * int(count * 0.2) + [4] * int(count * 0.4)
    random.shuffle(new_ns)

    ncfgs = []
    for hw, n, cin1, cin2, cin3, cin4 in zip(new_hws, new_ns, new_cins1, new_cins2, new_cins3, new_cins4):
        cins = [cin1, cin2, cin3, cin4]
        onehot = [1] * n + [0] * (4 - n)
        onehot_cins = [x * y for x, y in zip(onehot, cins)]
        c = {
            'HW': hw,
            'CIN1': onehot_cins[0],
            'CIN2': onehot_cins[1],
            'CIN3': onehot_cins[2],
            'CIN4': onehot_cins[3]
        }
        ncfgs.append(c)
    return ncfgs
