# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import numpy as np
import numpy as np
import scipy.interpolate as interpolate


def read_conv_zoo(filename = "data_sampler/prior_zoo/modelzoo_convs.csv"):
    f = open(filename,'r')
    i = 0
    hws = []
    cins = []
    couts = []
    ks = []
    X = []
    groups = []
    strides = []        
    while True:
            line = f.readline()
            if not line:
                break 
            if i > 0:
                content = line.strip().split(',')
                modelname = content[0]
                hw = int(content[1])
                cin = int(content[3])
                cout = int(content[4])
                k = int(content[5])
                s = int(content[6])
                group = int(content[7])
                strides.append(s)
                hws.append(hw)
                cins.append(cin)
                couts.append(cout)
                ks.append(k)
                groups.append(group)
                X.append((hw,cin,cout,k,s,group))
            i += 1
    return hws,cins,couts,ks,groups,strides

def read_dwconv_zoo(filename="data_sampler/prior_zoo/modelzoo_dwconvs.csv"):
    f = open(filename,'r')
    i = 0
    hws = []
    cs = []
    ks = []
    strides = []
    while True:
        line = f.readline()
        if not line:
            break 
        if i > 0:
            content = line.strip().split(',')
            hw = int(content[1])
            cin = int(content[3])
            k = int(content[5])
            s = int(content[6])
            hws.append(hw)
            cs.append(cin)
            ks.append(k)
            strides.append(s)
        i += 1
    return hws,cs,ks,strides


def read_fc_zoo(filename="data_sampler/prior_zoo/modelzoo_fcs.csv"):
    f = open(filename,'r')
    cins = []
    couts = []
    i = 0
    while True:
        line = f.readline()
        if not line:
            break
        content  =  line.strip().split(',')
        if i > 0:
            cin = int(content[1])
            cout = int(content[2])
            cins.append(cin)
            couts.append(cout)
        i += 1
    return cins,couts


def read_pool_zoo(filename="data_sampler/prior_zoo/modelzoo_poolings.csv"):
    cins = []
    couts = []
    hws = []
    ks = []
    strides = []
    i = 0 
    f = open(filename,'r')
    while True:
        line = f.readline()
        if not line:
            break 
        if i > 0:
            content = line.strip().split(',') # input_h,input_w,cin,cout,ks,stride
            inputh = int(content[1])
            cin = int(content[3])
            cout = int(content[4])
            k = int(content[5])
            s = int(content[6])
            cins.append(cin)
            # couts.append(cout)
            hws.append(inputh)
            ks.append(k)
            strides.append(s)
        i += 1 
    return hws,cins,ks,strides


def inverse_transform_sampling(data, n_bins = 40, n_samples = 1000):
    ''' calculate inversed cdf, for sampling by possibility
    '''
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    r = np.random.rand(n_samples)
    #print(r)
    da = inv_cdf(r)
    nda = [int(x) for x in da]
    return nda


def sample_based_on_distribution(data,count):
    ''' use data to calculate a inversed cdf, and sample `count` data from such distribution
    '''
    da = inverse_transform_sampling(data,n_samples = count)
    return da


def data_clean(data,cdata):
    ''' convert sampled data to valid configuration, e.g.,: kernel size=1, 3, 5, 7
    '''
    newlist = []
    for da in cdata:
        value = [abs(da-x) for x in data]
        newlist.append(value)

    newlist = list(np.asarray(newlist).T)    
    cda = [list(d).index(min(d)) for d in newlist]
    redata = [cdata[x] for x in cda]
    return redata