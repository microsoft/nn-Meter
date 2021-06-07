# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import pickle,os
def get_hswish_feature(features):
    nfs=[]
    for fe in features:
        [inputh,cin,cout,ks,s,flops,params]=fe
        if s==2:
            nfe=[inputh/2,cout]
        else:
            nfe=[inputh,cout]
        nfs.append(nfe)
    return nfs

def get_kernel_name(optype):
    if 'conv' in optype and 'dwconv' not in optype:
        optype='conv-bn-relu'    
    if 'dwconv' in optype:
        optype='dwconv-bn-relu'
    if optype=='fc-relu':
        optype='fc'
    if optype=='max-pool':
        optype='maxpool'
    if optype=='avg-pool':
        optype='avgpool'
    if optype in ['global-pool','gap']:
        optype='global-avgpool'
    if optype=='channel_shuffle':
        optype='channelshuffle'
    if optype in ['bn-relu']:
        optype='bnrelu'
    if optype in ['add-relu']:
        optype='addrelu'
        
    if optype in ['SE','SE-relu','se','se-relu']:
        optype='se'

    return optype
    

    