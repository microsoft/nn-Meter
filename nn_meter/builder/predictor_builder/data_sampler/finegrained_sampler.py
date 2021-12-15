# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import numpy as np 
import random
import copy


def sample(mind, maxd, D):
    '''sample D data from a range [mind, maxd] #TODO: make sure if there is sth wrong
    '''
    if maxd - mind <= D:
        cins = list(range(mind, maxd))
        random.shuffle(cins)
        return cins
    else:
        return random.sample(range(mind, maxd), D)


def sample_finegrained(cin, cout, D): 
    '''fine-grained sample D data in the cin and cout dimensions, respectively
    '''
    cins = sample(int(cin * 0.5), int(cin * 1.2), D)
    couts = sample(int(cout * 0.5), int(cout * 1.2), D)
    return cins, couts


def finegrained_sampling_conv(cfgs, count):
    ncfgs = []
    for cfg in cfgs:
        cins, couts = sample_finegrained(cfg['CIN'], cfg['COUT'], count)
        for i in range(min(len(cins), len(couts))):
                c = {}
                c['CIN'] = cins[i]
                c['COUT'] = couts[i] 
                c['HW'] = cfg['HW']
                c['STRIDE'] = cfg['STRIDE']
                c['KERNEL_SIZE'] = cfg['KERNEL_SIZE']
                ncfgs.append(c)
    return ncfgs


def finegrained_sampling_dwconv(cfgs, count):
    ncfgs = []
    for cfg in cfgs:
        cins, couts = sample_finegrained(cfg['CIN'], cfg['CIN'], count)
        for i in range(len(cins)):
                c = {}
                c['CIN'] = cins[i]
                c['COUT'] = couts[i] 
                c['HW'] = cfg['HW']
                c['STRIDE'] = cfg['STRIDE']
                c['KERNEL_SIZE'] = cfg['KERNEL_SIZE']
                ncfgs.append(c)
    return ncfgs


def finegrained_sampling_fc(cfgs, count):
    ncfgs = []
    for cfg in cfgs:
        cins, couts = sample_finegrained(cfg['CIN'], cfg['COUT'], count)
        for i in range(min(len(cins), len(couts))):
                c = {}
                c['CIN'] = cins[i]
                c['COUT'] = couts[i] 
                ncfgs.append(c)
    return ncfgs


def finegrained_sampling_CIN(cfgs, count):
    ncfgs = []
    for cfg in cfgs:
        cins, _ = sample_finegrained(cfg['CIN'], cfg['CIN'], count)
        for i in range(len(cins)):
            c = {}
            c['CIN'] = cins[i]
            c['HW'] = cfg['HW']
            ncfgs.append(c)
    return ncfgs


def finegrained_sampling_CIN_odd(cfgs, count):
    ''' for some kernels, split/se/channelshuffle, only odd values are valid
    '''
    ncfgs = []
    for cfg in cfgs:
        cins, _ = sample_finegrained(cfg['CIN'], cfg['CIN'], count)
        for i in range(len(cins)):
            nc = cins[i]
            if cins[i] % 2 != 0:
                nc = cins[i] + 1 
            c = {}
            c['CIN'] = nc
            c['HW'] = cfg['HW']
            ncfgs.append(c)
    return ncfgs


def finegrained_sampling_concat(cfgs, count):
    ncfgs = []
    for cfg in cfgs:
        ncins = []
        ns = []
        for cin in cfg['CINS']:
            cins, _ = sample_finegrained(cin, cin, count)
            ns.append(len(cins))
            ncins.append(cins)
        for j in range(min(ns)):
            c = {}
            c['HW'] = cfg['HW']
            c['CINS'] = []
            for i in range(len(ncins)):
                c['CINS'].append(ncins[i][j])
            ncfgs.append(c)
    return ncfgs
