import numpy as np
import random
import warnings
import pandas as pd
import scipy.stats as st
import statsmodels as sm
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate


'''
read_conv(), read_dwconv(), read_fc(), read_pool(): we collect configuration values from existing model zoo for each kernel, and store them in prior/xxx.csv,
the four functions read the csvs
'''

def read_conv(filename="data_sampler/prior/modelzoo_convs.csv"):
    f=open(filename,'r')
    i=0
    hws=[]
    cins=[]
    couts=[]
    ks=[]
    X=[]
    groups=[]
    strides=[]        
    while True:
            line=f.readline()
            if not line:
                break 
            if i>0:
                content=line.strip().split(',')
                modelname=content[0]
                hw=int(content[1])
                cin=int(content[3])
                cout=int(content[4])
                k=int(content[5])
                s=int(content[6])
                group=int(content[7])
                strides.append(s)
                hws.append(hw)
                cins.append(cin)
                couts.append(cout)
                ks.append(k)
                groups.append(group)
                X.append((hw,cin,cout,k,s,group))
            i+=1
    return hws,cins,couts,ks,groups,strides
def read_dwconv(filename="data_sampler/prior/modelzoo_dwconvs.csv"):
    f=open(filename,'r')
    i=0
    hws=[]
    cs=[]
    ks=[]
    strides=[]
    while True:
        line=f.readline()
        if not line:
            break 
        if i>0:
            content=line.strip().split(',')
            hw=int(content[1])
            cin=int(content[3])
            k=int(content[5])
            s=int(content[6])
            hws.append(hw)
            cs.append(cin)
            ks.append(k)
            strides.append(s)
        i+=1
    return hws,cs,ks,strides
def read_fc(filename="data_sampler/prior/modelzoo_fcs.csv"):
    f=open(filename,'r')
    cins=[]
    couts=[]
    i=0
    while True:
        line=f.readline()
        if not line:
            break
        content=line.strip().split(',')
        if i>0:
            cin=int(content[1])
            cout=int(content[2])
            cins.append(cin)
            couts.append(cout)

        i+=1
    return cins,couts
def read_pool(filename="data_sampler/prior/modelzoo_poolings.csv"):
    cins=[]
    couts=[]
    hws=[]
    ks=[]
    strides=[]
    i=0 
    f=open(filename,'r')
    while True:
        line=f.readline()
        if not line:
            break 
        if i>0:
            content=line.strip().split(',') #input_h,input_w,cin,cout,ks,stride
            inputh=int(content[1])
            cin=int(content[3])
            cout=int(content[4])
            k=int(content[5])
            s=int(content[6])
            cins.append(cin)
            #couts.append(cout)
            hws.append(inputh)
            ks.append(k)
            strides.append(s)
        i+=1 
    return hws,cins,ks,strides

###calculate inversed cdf, for sampling by possibility
def inverse_transform_sampling(data, n_bins=40, n_samples=1000):
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    r = np.random.rand(n_samples)
    #print(r)
    da= inv_cdf(r)
    nda=[int(x) for x in da]
    return nda

### use data to calculate a inversed cdf, and sample `count` data from such distribution
def sample_based_on_distribution(data,count):
    da=inverse_transform_sampling(data,n_samples=count)
    return da
'''
the followings are sampling functions for each kernels
'''



def data_clean(data,cdata):  ## convert sampled data to valid configuration, e.g.,: kernel size=1,3,5,7
    newlist=[]
    for da in cdata:
        value=[abs(da-x) for x in data]
        newlist.append(value)

    newlist=list(np.asarray(newlist).T)    
    cda=[list(d).index(min(d)) for d in newlist]
    redata=[cdata[x] for x in cda]
    return redata


def sampling_conv(count):
    hws,cins,couts,ks,groups,strides=read_conv()
    newcins=sample_based_on_distribution(cins,count)     
    newcouts=sample_based_on_distribution(couts,count)

    ### 70% of sampled data are totally from prior distribution
    count1=int(count*0.7)
    newhws=sample_based_on_distribution(hws,count1)
    newks=sample_based_on_distribution(ks,count1)
    newstrides=sample_based_on_distribution(strides,count1)
    
    newks=data_clean(newks,[1,3,5,7])
    newstrides=data_clean(newstrides,[1,2,4])
    newhws=data_clean(newhws,[1,3,7,8,13,14,27,28,32,56,112,224])
    
    ## since conv is the largest and most-challenging kernel, we add some frequently used configuration values
    newhws.extend([112]*int((count-count1)*0.2)+[56]*int((count-count1)*0.4)+[28]*int((count-count1)*0.4))  ## frequent settings
    newks.extend([5]*int((count-count1)*0.4)+[7]*int((count-count1)*0.6))## frequent settings
    newstrides.extend([2]*int((count-count1)*0.4)+[1]*int((count-count1)*0.6))## frequent settings
    random.shuffle(newhws)
    random.shuffle(newstrides)
    random.shuffle(newks)
    print(len(newcins),len(newstrides),len(newks),len(newhws))
    return newcins,newcouts,newhws,newks,newstrides

def sampling_conv_random(count):
    hws=[224,112,56,32,28,27,14,13,8,7,1]
    ks=[1,3,5,7]
    s=[1,2,4]
    
    cins=list(range(3,2160))
    couts=list(range(16,2048))
    newhws=random.sample(hws*int(count/len(hws))*10,count)
    newks=random.sample(ks*int(count/len(ks)*10),count)
    newstrides=random.sample(s*int(count/len(s)*10),count)
    newcins=random.sample(cins*10,count)
    newcouts=random.sample(couts*18,count)
    random.shuffle(newcins)
    random.shuffle(newcouts)
    keys=[]
    for index in range(len(newhws)):
        key='_'.join(str(x) for x in[newhws[index],newks[index],newstrides[index],newcins[index],newcouts[index]])
        if not key in keys:
            keys.append(key)
    print(len(keys))
   
    
    return newcins,newcouts,newhws,newks,newstrides




def sampling_dwconv(count):
    hws,cs,ks,strides=read_dwconv()
    newcs=sample_based_on_distribution(cs,count)
   
    count1=int(count*0.8)
    newhws=sample_based_on_distribution(hws,count1)
    newks=sample_based_on_distribution(ks,count1)
    newstrides=sample_based_on_distribution(strides,count1)
    
    newhws=data_clean(newhws,[1,3,7,14,28,56,112,224])
    newks=data_clean(newks,[1,3,5,7])
    newstrides=data_clean(newstrides,[1,2])
    
    newhws.extend([112]*int((count-count1)*0.4)+[56]*int((count-count1)*0.4)+[28]*int((count-count1)*0.2))  
    newks.extend([5]*int((count-count1)*0.4)+[7]*int((count-count1)*0.6))
    newstrides.extend([2]*int((count-count1)*0.5)+[1]*int((count-count1)*0.5))
    random.shuffle(newhws)
    random.shuffle(newks)
    random.shuffle(newstrides)
    print(len(newcs),len(newhws),len(newks),len(newstrides))
   
    return newcs,newhws,newks,newstrides

def sampling_addrelu(count,ratio=1.8):
    hws,cins,couts,ks,groups,strides=read_conv()    
    newcs=sample_based_on_distribution(cins,count)
    newcs=[int(x*ratio) for x in newcs]
    newhws=[14]*int(count*0.4)+[7]*int(count*0.4)+[28]*int(count*0.2)
    random.shuffle(newhws)
    print(newcs)
    return newhws,newcs,newcs


def sampling_fc(count,fix_cout=1000):
    cins,couts=read_fc()
    newcins=sample_based_on_distribution(cins,count)
    if not fix_cout:
        newcouts=sample_based_on_distribution(couts,count)
    else:
        newcouts=[fix_cout]*count
    return newcins,newcouts

def sampling_pooling(count):
    hws,cins,ks,strides=read_pool()
    newcins=sample_based_on_distribution(cins,count)
    newhws=sample_based_on_distribution(hws,count)
    newks=sample_based_on_distribution(ks,count)
    newstrides=sample_based_on_distribution(strides,count)

    newhws=data_clean(newhws,[14,28,56,112,224])
    newks=data_clean(newks,[3])
    newstrides=data_clean(newstrides,[2])
    return newhws,newcins,newks,newstrides

def sampling_concats(count):
    hws,cins,couts,ks,groups,strides=read_conv()
    newcins1=sample_based_on_distribution(cins,count)
    newcins2=sample_based_on_distribution(cins,count)
    newcins3=sample_based_on_distribution(cins,count)
    newcins4=sample_based_on_distribution(cins,count)
     
   
    newhws=sample_based_on_distribution(hws,count)
    newhws=data_clean(newhws,[7,14,28,56])  ## current normals
    newns=[2]*int(count*0.4)+[3]*int(count*0.2)+[4]*int(count*0.4)
    if len(newns)<count:
        da=[2]*(count-len(newns))
        newns.extend(da)
    random.shuffle(newns)
    return newhws,newns,newcins1,newcins2,newcins3,newcins4
    




