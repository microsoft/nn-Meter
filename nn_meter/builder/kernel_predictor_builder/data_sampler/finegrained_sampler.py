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
                'STRIDES': cfg['POOL_STRIDES'],
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


def finegrained_sampling_hw_cin_even(cfgs, count):
    ''' sampling configs for kernels with hw and cin (only even values) parameter, in case for split / se / channelshuffle
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
    Returned params include: (hw, cin1, cin2, cin3, cin4). Note that we only sample (num of cin) = 2, 3, 4, 
    (cin1, cin2, cin3, cin4) is one-hot vector with unused input channel set as 0.
    '''
    ncfgs = []
    for cfg in cfgs:
        ncins, total_cins = [], []
        for cin in [cfg['CIN1'], cfg['CIN2'], cfg['CIN3'], cfg['CIN4']]:
            if cin == 0:
                total_cins.append([0] * count)
                continue
            cins = sample_in_range(int(cin * 0.5), int(cin * 1.2), count)
            ncins.append(len(cins))
            total_cins.append(cins)
        for j in range(min(ncins)):
            cins = [total_cins[i][j] for i in range(4)]
            c = {
                'HW': cfg['HW'],
                'CIN1': cins[0],
                'CIN2': cins[1],
                'CIN3': cins[2],
                'CIN4': cins[3]
            }
            ncfgs.append(c)
    return ncfgs
