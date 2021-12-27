# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import numpy as np
import scipy.interpolate as interpolate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def read_conv_zoo(filename = "modelzoo_convs.csv"):
    filename = os.path.join(BASE_DIR, "prior_zoo", filename)
    f = open(filename,'r')
    i = 0
    hws, cins, couts, ks, groups, strides = [], [], [], [], [], []   
    while True:
        line = f.readline()
        if not line:
            break 
        if i > 0:
            # model, input_h, input_w, cin, cout, ks, stride, groups
            content = line.strip().split(',')
            hws.append(int(content[1]))
            cins.append(int(content[3]))
            couts.append(int(content[4]))
            ks.append(int(content[5]))
            strides.append(int(content[6]))
            groups.append(int(content[7]))
        i += 1
    return hws, cins, couts, ks, groups, strides


def read_dwconv_zoo(filename = "modelzoo_dwconvs.csv"):
    filename = os.path.join(BASE_DIR, "prior_zoo", filename)
    f = open(filename,'r')
    i = 0
    hws, cins, ks, strides = [], [], [], []
    while True:
        line = f.readline()
        if not line:
            break 
        if i > 0:
            # model, input_h, input_w, cin, cout, ks, stride, groups
            content = line.strip().split(',')
            hws.append(int(content[1]))
            cins.append(int(content[3]))
            ks.append(int(content[5]))
            strides.append(int(content[6]))
        i += 1
    return hws, cins, ks, strides


def read_fc_zoo(filename = "modelzoo_fcs.csv"):
    filename = os.path.join(BASE_DIR, "prior_zoo", filename)
    f = open(filename,'r')
    cins, couts = [], []
    i = 0
    while True:
        line = f.readline()
        if not line:
            break
        if i > 0:
            # model, cin, cout
            content = line.strip().split(',')
            cins.append(int(content[1]))
            couts.append(int(content[2]))
        i += 1
    return cins, couts


def read_pool_zoo(filename = "modelzoo_poolings.csv"):
    filename = os.path.join(BASE_DIR, "prior_zoo", filename)
    cins, hws, ks, strides = [], [], [], []
    i = 0 
    f = open(filename,'r')
    while True:
        line = f.readline()
        if not line:
            break 
        if i > 0:
            # model, input_h, input_w, cin, cout, ks, stride
            content = line.strip().split(',')
            hws.append(int(content[1]))
            cins.append(int(content[3]))
            ks.append(int(content[5]))
            strides.append(int(content[6]))
        i += 1 
    return hws, cins, ks, strides


def inverse_transform_sampling(data, n_bins = 40, n_samples = 1000):
    ''' calculate inversed cdf, for sampling by possibility
    '''
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
