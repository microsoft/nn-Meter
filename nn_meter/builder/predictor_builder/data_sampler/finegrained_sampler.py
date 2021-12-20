# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import numpy as np 
import random
import copy


def sample_in_range(mind, maxd, sample_num):
    '''sample D data from a range [mind, maxd)
    '''
    # if the sample_num is bigger than sample population, we only keep the number of population to avoid repetition
    if maxd - mind <= sample_num:
        data = list(range(mind, maxd))
        random.shuffle(data)
        return data
    else:
        return random.sample(range(mind, maxd), sample_num)


def sample_cin_cout(cin, cout, sample_num): 
    '''fine-grained sample D data in the cin and cout dimensions, respectively
    '''
    cins = sample_in_range(int(cin * 0.5), int(cin * 1.2), sample_num)
    couts = sample_in_range(int(cout * 0.5), int(cout * 1.2), sample_num)
    l = min(len(cins), len(couts)) # align the length of cins and couts
    cins, couts = cins[:l], couts[:l]
    return cins, couts


def finegrained_sampling_conv(cfgs, count):
    ncfgs = []
    for cfg in cfgs:
        cins, couts = sample_cin_cout(cfg['CIN'], cfg['COUT'], count)
        for cin, cout in zip(cins, couts):
            c = {
                'CIN': cin,
                'COUT': cout,
                'HW': cfg['HW'],
                'STRIDE': cfg['STRIDE'],
                'KERNEL_SIZE': cfg['KERNEL_SIZE']
            }
            ncfgs.append(c)
    return ncfgs


def finegrained_sampling_dwconv(cfgs, count):
    ncfgs = []
    for cfg in cfgs:
        cins, couts = sample_cin_cout(cfg['CIN'], cfg['CIN'], count)
        for cin, cout in zip(cins, couts):
            c = {
                'CIN': cin,
                'COUT': cout,
                'HW': cfg['HW'],
                'STRIDE': cfg['STRIDE'],
                'KERNEL_SIZE': cfg['KERNEL_SIZE']
            }
            ncfgs.append(c)
    return ncfgs


def finegrained_sampling_fc(cfgs, count):
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


def finegrained_sampling_CIN(cfgs, count):
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


def finegrained_sampling_CIN_odd(cfgs, count):
    ''' for some kernels, split/se/channelshuffle, only odd values are valid
    '''
    ncfgs = []
    for cfg in cfgs:
        cins = sample_in_range(int(cfg['CIN'] * 0.5), int(cfg['CIN'] * 1.2), count)
        for cin in cins:
            nc = cin if cin % 2 else cin + 1
            c = {
                'CIN': nc,
                'HW': cfg['HW']
            }
            ncfgs.append(c)
    return ncfgs


def finegrained_sampling_concat(cfgs, count):
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
                'CINS': [total_cins[i][j] for i in range(len(cfg['CINS']))]
            }
            ncfgs.append(c)
    return ncfgs
