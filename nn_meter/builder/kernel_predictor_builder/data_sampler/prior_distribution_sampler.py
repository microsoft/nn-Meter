# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import random
import numpy as np
from nn_meter.utils import get_conv_flop_params, get_dwconv_flop_params, get_fc_flop_params
from .prior_config_lib.utils import *
from nn_meter.builder.utils import make_divisible

_conv_hw_candidate = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 64,
    66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98,
    100, 102, 104, 106, 108, 110, 112, 128, 132, 136, 140, 144, 148, 152,
    156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208,
    212, 216, 220, 224
]

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
    new_hws = list(map(int, np.random.choice(_conv_hw_candidate, size=count1, replace=True)))
    new_kernel_sizes = sample_based_on_distribution(kernel_sizes, count1)
    new_strides = sample_based_on_distribution(strides, count1)

    new_kernel_sizes = data_validation(new_kernel_sizes, [1, 3, 5, 7])
    new_strides = data_validation(new_strides, [1, 2, 4])

    # since conv is the largest and most-challenging kernel, we add some frequently used configuration values
    new_hws.extend([112] * int((count - count1) * 0.2) + [56] * int((count - count1) * 0.4) + [28] * int((count - count1) * 0.4)) # frequent settings
    new_kernel_sizes.extend([5] * int((count - count1) * 0.4) + [7] * int((count - count1) * 0.6)) # frequent settings
    new_strides.extend([2] * int((count - count1) * 0.4) + [1] * int((count - count1) * 0.6)) # frequent settings
    random.shuffle(new_hws)
    random.shuffle(new_strides)
    random.shuffle(new_kernel_sizes)

    ncfgs = []
    nparams = [] # calculate the number of parameters for configs sort
    for hw, cin, cout, kernel_size, stride in zip(new_hws, new_cins, new_couts, new_kernel_sizes, new_strides):
        c = {
            'HW': hw,
            'CIN': cin,
            'COUT': cout,
            'KERNEL_SIZE': kernel_size,
            'STRIDES': stride,
        }
        ncfgs.append(c)
        nparams.append(get_conv_flop_params(hw, cin, cout, kernel_size, stride))

    # sort all sampling configs by number of parameters, from the smallest to the largest
    # the procedure is for better profiling
    ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]

    return ncfgs


def sampling_conv_ofa(count):
    ''' 
    Sampling configs for conv kernels based on conv_zoo, which contains configuration values from existing model zoo for conv kernel. 
    The values are stored in prior_config_lib/conv.csv.
    Returned params include: (hw, cin, cout, kernel_size, strides)
    '''
    hws, cins, couts, kernel_sizes, strides = read_conv_zoo(filename='conv_ofa.csv')
    hw_candidate = [3, 5, 6, 7, 10, 11, 12, 13, 14, 20, 22, 24, 26, 28, 40, 44,
                    48, 52, 56, 80, 88, 96, 104, 112, 160, 176, 192, 208, 224]
    cin_candidate = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 128,
                     136, 160, 192, 624, 768, 832, 960, 1024, 1152, 1280, 1536]

    # 50% of sampled data are from prior distribution
    count1 = int(count * 0.5)
    new_hws = sample_based_on_distribution(hws, count1)
    new_kernel_sizes = sample_based_on_distribution(kernel_sizes, count1)
    new_strides = sample_based_on_distribution(strides, count1)
    new_cins = sample_based_on_distribution(cins, count1)
    new_couts = sample_based_on_distribution(couts, count1)

    new_kernel_sizes = data_validation(new_kernel_sizes, [1, 3, 5, 7])
    new_strides = data_validation(new_strides, [1, 2])
    new_hws = data_validation(new_hws, hw_candidate)
    random.shuffle(new_hws)
    random.shuffle(new_strides)
    random.shuffle(new_kernel_sizes)

    # since conv is the largest and most-challenging kernel, we add some frequently used configuration values
    count2 = count - count1
    count2_0 = count2 // 4 # hw is fixed to 224 and cin is fixed to 3
    count2_1 = count2 // 4
    count2_2 = count2 // 4
    count2_3 = count2 - count2_0 - count2_1 - count2_2
    # count2_0: hw is fixed to 224 and cin is fixed to 3
    new_hws.extend([224 for _ in range(count2_0)]) # frequent settings
    new_kernel_sizes.extend(list(map(int, np.random.choice([3, 5, 7], size=count2_0, replace=True)))) # frequent settings
    new_strides.extend(list(map(int, np.random.choice([1, 2], size=count2_0, replace=True)))) # frequent settings
    cins_extend0 = [3 for _ in range(count2_0)]
    cout_extend0 = [make_divisible(3 * (np.random.rand() * 3 + 3))] * count2_0
    new_cins.extend(cins_extend0)
    new_couts.extend(cout_extend0)

    # count2_1: input channel: cin; output channel: cin * expand_ratio; kernel_size: 1; strides: 1
    new_hws.extend(list(map(int, np.random.choice(hw_candidate, size=(count2 - count2_0), replace=True)))) # frequent settings
    new_kernel_sizes.extend([1] * (count2_1 + count2_2)) # frequent settings
    new_strides.extend([1] * (count2_1 + count2_2)) # frequent settings
    cins_extend1 = list(map(int, np.random.choice(cin_candidate, size=count2_1, replace=True)))
    cout_extend1 = [make_divisible(cin * (np.random.rand() * 3 + 3)) for cin in cins_extend1] # cout = cin * expand_ratio
    new_cins.extend(cins_extend1)
    new_couts.extend(cout_extend1)

    # count2_2: input channel: cin; output channel: cin / expand_ratio; kernel_size: 1; strides: 1
    cins_extend2 = list(map(int, np.random.choice(cin_candidate, size=count2_2, replace=True)))
    cout_extend2 = [make_divisible(cin / (np.random.rand() * 3 + 3)) for cin in cins_extend2] # cout = cin * expand_ratio
    new_cins.extend(cins_extend2)
    new_couts.extend(cout_extend2)

    # count2_3: input channel: cin * expand_ratio; output channel: cin * expand_ratio
    new_kernel_sizes.extend(list(map(int, np.random.choice([3, 5, 7], size=count2_3, replace=True)))) # frequent settings
    new_strides.extend(list(map(int, np.random.choice([1, 2], size=count2_3, replace=True)))) # frequent settings
    c_extend3 = [make_divisible(cin / (np.random.rand() * 3 + 3)) for cin in 
                    list(map(int, np.random.choice(cin_candidate, size=count2_3, replace=True)))]
    new_cins.extend(c_extend3)
    new_couts.extend(c_extend3)

    ncfgs = []
    for hw, cin, cout, kernel_size, stride in zip(new_hws, new_cins, new_couts, new_kernel_sizes, new_strides):
        c = {
            'HW': hw,
            'CIN': make_divisible(cin),
            'COUT': make_divisible(cout),
            'KERNEL_SIZE': kernel_size,
            'STRIDES': stride,
        }
        if c not in ncfgs:
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
    nparams = [] # calculate the number of parameters for configs sort
    for hw, cin, cout, kernel_size, stride in zip(new_hws, new_cins, new_couts, new_kernel_sizes, new_strides):
        c = {
            'HW': hw,
            'CIN': cin,
            'COUT': cout,
            'KERNEL_SIZE': kernel_size,
            'STRIDES': stride,
        }
        ncfgs.append(c)
        nparams.append(get_conv_flop_params(hw, cin, cout, kernel_size, stride))

    # sort all sampling configs by number of parameters, from the smallest to the largest
    # the procedure is for better profiling
    ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]

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
    new_hws = sample_based_on_distribution(hws, count1)
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
    nparams = [] # calculate the number of parameters for configs sort
    for hw, cin, kernel_size, stride in zip(new_hws, new_cins, new_kernel_sizes, new_strides):
        c = {
            'HW': hw,
            'CIN': cin,
            'KERNEL_SIZE': kernel_size,
            'STRIDES': stride,
        }
        ncfgs.append(c)
        nparams.append(get_dwconv_flop_params(hw, cin, kernel_size, stride))

    # sort all sampling configs by number of parameters, from the smallest to the largest
    # the procedure is for better profiling
    ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]

    return ncfgs


def sampling_dwconv_ofa(count):
    ''' 
    Sampling configs for dwconv kernels based on dwconv zoo, which contains configuration values from existing model zoo for dwconv kernel. 
    The values are stored in prior_config_lib/dwconv.csv.
    Returned params include: (hw, cin, kernel_size, strides)
    '''
    hw_candidate = [3, 5, 6, 7, 10, 11, 12, 13, 14, 20, 22, 24, 26, 28, 40, 44,
                    48, 52, 56, 80, 88, 96, 104, 112, 160, 176, 192, 208, 224]
    cin_candidate = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 128,
                     136, 160, 192, 624, 768, 832, 960, 1024, 1152, 1280, 1536]

    hws, cins, ks, strides = read_dwconv_zoo(filename='dwconv_ofa.csv')

    count1 = int(count * 0.7)
    # import pdb; pdb.set_trace()
    new_hws = sample_based_on_distribution(hws, count1)
    new_kernel_sizes = sample_based_on_distribution(ks, count1)
    new_strides = sample_based_on_distribution(strides, count1)
    new_cins = sample_based_on_distribution(cins, count1)

    new_hws = data_validation(new_hws, hw_candidate)
    new_kernel_sizes = data_validation(new_kernel_sizes, [1, 3, 5, 7])
    new_strides = data_validation(new_strides, [1, 2])

    count2 = count - count1
    new_hws.extend(list(map(int, np.random.choice(hw_candidate, size=count2, replace=True)))) # frequent settings
    new_kernel_sizes.extend(list(map(int, np.random.choice([3, 5, 7], size=count2, replace=True)))) # frequent settings
    new_strides.extend(list(map(int, np.random.choice([1, 2], size=count2, replace=True)))) # frequent settings
    new_cins.extend(list(map(int, np.random.choice(cin_candidate, size=count-len(new_cins), replace=True))))
    random.shuffle(new_hws)
    random.shuffle(new_kernel_sizes)
    random.shuffle(new_strides)
    random.shuffle(new_cins)

    ncfgs = []
    for hw, cin, kernel_size, stride in zip(new_hws, new_cins, new_kernel_sizes, new_strides):
        # print(hw, cin, kernel_size, stride)
        c = {
            'HW': hw,
            'CIN': make_divisible(cin),
            'KERNEL_SIZE': kernel_size,
            'STRIDES': stride,
        }
        if c not in ncfgs:
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
    nparams = [] # calculate the number of parameters for configs sort
    for cin, cout in zip(new_cins, new_couts):
        c = {
            'CIN': cin,
            'COUT': cout,
        }
        ncfgs.append(c)
        nparams.append(get_fc_flop_params(cin, cout))

    # sort all sampling configs by number of parameters, from the smallest to the largest
    # the procedure is for better profiling
    ncfgs = [x for x, _ in sorted(zip(ncfgs, nparams), key=lambda x: x[1])]

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


def sampling_hw_cin_ofa(count):
    ''' sampling configs for kernels with hw and cin parameter
    Returned params include: (hw, cin)
    '''
    hw_candidate = [3, 5, 6, 7, 10, 11, 12, 13, 14, 20, 22, 24, 26, 28, 40, 44,
                    48, 52, 56, 80, 88, 96, 104, 112, 160, 176, 192, 208, 224]
    hws, cins, _, _, _ = read_conv_zoo(filename='conv_ofa.csv')
    new_cins = sample_based_on_distribution(cins, count)
   
    count1 = int(count * 0.5)
    new_hws = sample_based_on_distribution(hws, count1)
    new_hws = data_validation(new_hws, hw_candidate)
    new_hws.extend([224] * int((count - count1) * 0.2) + 
                   list(map(int, np.random.choice(hw_candidate, size=int((count - count1) * 0.8), replace=True))))
    random.shuffle(new_hws)

    ncfgs = []
    for hw, cin in zip(new_hws, new_cins):
        c = {
            'HW': hw,
            'CIN': make_divisible(cin),
        }
        if c not in ncfgs:
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


def sampling_resnet_se(count):
    ''' sampling configs for kernels with mid_channel specified
    '''
    hws, cins, _, _, _ = read_conv_zoo()
    new_cins = sample_based_on_distribution(cins, count)

    count1 = int(count * 0.8)
    new_hws = sample_based_on_distribution(hws,count1)
    new_hws = data_validation(new_hws, _conv_hw_candidate)
    new_hws.extend([112] * int((count - count1) * 0.4) + [56] * int((count - count1) * 0.4) + [28] * int((count - count1) * 0.2))
    random.shuffle(new_hws)
    
    filename = os.path.join("/data/jiahang/working/nn-Meter/nn_meter/builder/kernel_predictor_builder/data_sampler/prior_config_lib/resnet_se.csv")
    se_df = pd.read_csv(filename)
    cins = se_df["cin"]
    couts = se_df["cout"]
    exp = se_df["exp"]

    ncfgs = []
    for hw, cin, cout, exp in zip(new_hws, cins, couts, exp):
        c = {
            'HW': hw,
            'CIN': make_divisible(cout * exp),
            'CMID': make_divisible(cin // 4)
        }
        ncfgs.append(c)
    return ncfgs


def sampling_hw_cin_even_ofa(count):
    ''' sampling configs for kernels with hw and cin (only even values) parameter, in case for split / se / channelshuffle
    Returned params include: (hw, cin)
    '''
    hw_candidate = [3, 5, 6, 7, 10, 11, 12, 13, 14, 20, 22, 24, 26, 28, 40, 44,
                    48, 52, 56, 80, 88, 96, 104, 112, 160, 176, 192, 208, 224]
    hws, cins, _, _, _ = read_conv_zoo(filename='conv_ofa.csv')
    new_cins = sample_based_on_distribution(cins, count)
   
    count1 = int(count * 0.5)
    new_hws = sample_based_on_distribution(hws, count1)
    new_hws = data_validation(new_hws, hw_candidate)
    new_hws.extend([224] * int((count - count1) * 0.2) + 
                   list(map(int, np.random.choice(hw_candidate, size=int((count - count1) * 0.8), replace=True))))
    random.shuffle(new_hws)

    ncfgs = []
    for hw, cin in zip(new_hws, new_cins):
        c = {
            'HW': hw,
            'CIN': make_divisible(cin + 1 if cin % 2 else cin),
        }
        if c not in ncfgs:
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
