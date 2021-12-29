# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import random


def sample_in_range(mind, maxd, sample_num):
    '''sample #sample_num data from a range [mind, maxd)
    '''
    # if the sample_num is bigger than sample population, we only keep the number of population to avoid repetition
    if maxd - mind <= sample_num:
        data = list(range(mind, maxd))
        random.shuffle(data)
        return data
    else:
        return random.sample(range(mind, maxd), sample_num)


def sample_cin_cout(cin, cout, sample_num): 
    '''fine-grained sample #sample_num data in the cin and cout dimensions, respectively
    '''
    cins = sample_in_range(int(cin * 0.5), int(cin * 1.2), sample_num)
    couts = sample_in_range(int(cout * 0.5), int(cout * 1.2), sample_num)
    l = min(len(cins), len(couts)) # align the length of cins and couts
    cins, couts = cins[:l], couts[:l]
    return cins, couts


def finegrained_sampling_conv(cfgs, count):
    ''' 
    Sampling configs for conv kernels
    Returned params include: (hw, cin, cout, kernel_size, strides)
    '''
    ncfgs = []
    for cfg in cfgs:
        cins, couts = sample_cin_cout(cfg['CIN'], cfg['COUT'], count)
        for cin, cout in zip(cins, couts):
            c = {
                'HW': cfg['HW'],
                'CIN': cin,
                'COUT': cout,
                'KERNEL_SIZE': cfg['KERNEL_SIZE'],
                'STRIDES': cfg['STRIDES'],
            }
            ncfgs.append(c)
    return ncfgs


def finegrained_sampling_dwconv(cfgs, count):
    ''' 
    Sampling configs for dwconv kernels
    Returned params include: (hw, cin, kernel_size, strides)
    '''
    ncfgs = []
    for cfg in cfgs:
        cins = sample_in_range(int(cfg['CIN'] * 0.5), int(cfg['CIN'] * 1.2), count)
        for cin in cins:
            c = {
                'HW': cfg['HW'],
                'CIN': cin,
                'KERNEL_SIZE': cfg['KERNEL_SIZE'],
                'STRIDES': cfg['STRIDES'],
            }
            ncfgs.append(c)
    return ncfgs


def finegrained_sampling_fc(cfgs, count):
    '''
    Sampling configs for fc kernels
    Returned params include: (cin, cout)
    '''
    ncfgs = []
    for cfg in cfgs:
        cins, couts = sample_cin_cout(cfg['CIN'], cfg['COUT'], count)
        for cin, cout in zip(cins, couts):
            c = {
                'CIN': cin,
                'COUT': cout
            }
            ncfgs.append(c)
    return ncfgs


def finegrained_sampling_pooling(cfgs, count):
    '''
    Sampling configs for pooling kernels
    Returned params include: (hw, cin, kernel_size, pool_strides)
    '''
    ncfgs = []
    for cfg in cfgs:
        cins = sample_in_range(int(cfg['CIN'] * 0.5), int(cfg['CIN'] * 1.2), count)
        for cin in cins:
            c = {
                'HW': cfg['HW'],
                'CIN': cin,
                'KERNEL_SIZE': cfg['KERNEL_SIZE'],
                'STRIDES': cfg['STRIDES'],
            }
            ncfgs.append(c)
    return ncfgs


def finegrained_sampling_hw_cin(cfgs, count):
    ''' sampling configs for kernels with hw and cin parameter
    Returned params include: (hw, cin)
    '''
    ncfgs = []
    for cfg in cfgs:
        cins = sample_in_range(int(cfg['CIN'] * 0.5), int(cfg['CIN'] * 1.2), count)
        for cin in cins:
            c = {
                'CIN': cin,
                'HW': cfg['HW']
            }
            ncfgs.append(c)
    return ncfgs


def finegrained_sampling_hw_cin_odd(cfgs, count):
    ''' sampling configs for kernels with hw and cin (only odd values) parameter, in case for split / se / channelshuffle
    Returned params include: (hw, cin)
    '''
    ncfgs = []
    for cfg in cfgs:
        cins = sample_in_range(int(cfg['CIN'] * 0.5), int(cfg['CIN'] * 1.2), count)
        for cin in cins:
            c = {
                'CIN': cin + 1 if cin % 2 else cin,
                'HW': cfg['HW']
            }
            ncfgs.append(c)
    return ncfgs


def finegrained_sampling_concats(cfgs, count):
    ''' sampling functions for concat kernel
    Returned params include: (hw, ns, cin1, cin2, cin3, cin4), ns are in [2, 4]
    '''
    ncfgs = []
    for cfg in cfgs:
        ncins, total_cins = [], []
        for cin in cfg['CINS']:
            cins = sample_in_range(int(cin * 0.5), int(cin * 1.2), count)
            ncins.append(len(cins))
            total_cins.append(cins)
        for j in range(min(ncins)):
            c = {
                'HW': cfg['HW'],
                'NS': cfg['NS'],
                'CINS': [total_cins[i][j] for i in range(len(cfg['CINS']))]
            }
            ncfgs.append(c)
    return ncfgs


def finegrained_kernel_sampling(kernel_type, configs, count):
    """
    return the list of sampled data configurations in finegrained sampling phase

    @params

    kernel_type: identical kernel name 
    count: int 
    for each large-error-data-point, we sample `count` more data around it.
    cfgs: list
    each item in the list represent a large-error-data-point. each item is a dictionary, storing the configuration

    """
    from .utils import config_for_kernel
    assert kernel_type in config_for_kernel.keys(), f"Not supported kernel type: {kernel_type}. Supported type includes {config_for_kernel.keys()}."

    if "conv" in kernel_type:
        return finegrained_sampling_conv(configs, count)
    elif "dwconv" in kernel_type:
        return finegrained_sampling_dwconv(configs, count)
    elif kernel_type in ['maxpool_block', 'avgpool_block']:
        return finegrained_sampling_pooling(count, fix_ks=3, fix_stride=1)
    if kernel_type == 'fc_block':
        return finegrained_sampling_fc(configs, count)
    if kernel_type in ['concat_block']:
        return finegrained_sampling_concats(configs, count)
    if kernel_type in ['split_block', 'se_block', 'channel_shuffle_block']:
        return finegrained_sampling_hw_cin_odd(configs, count)
    else: # 'hswish_block', 'bn_relu', 'bn_block', 'relu_block', 'add_relu', 'add_blocks'
        return finegrained_sampling_hw_cin(configs, count)