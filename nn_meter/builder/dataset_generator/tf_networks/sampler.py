# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import random

def sampling(list):
        return random.choice(list)


def get_channel(cmin, cmax, minstep):
    c = list(range(cmin, cmax + 1, minstep))
    return c


def sample_other_params(cfg, dicts):
    
    for item in cfg:
            data = cfg[item]
            if isinstance(data, list)  ==  True:
                    da = sampling(data)
                    #print(item, da)
                    dicts[item] = da
                    if item  ==  'HW' and da  ==  224:  ### remove out illegal samples
                        cin = 3
                        dicts['CIN'] = 3 
                    if item  ==  'GW':
                        if cin%da !=  0 : ## group number must be divided by cin/cout
                            dicts[item] = 1
    return dicts


class BasePatternSampler:
    def __init__(self, cfg, snum, dw = False):
        self.cfg = cfg 
       
        self.samples = []
        #snum = cfg['SAMPLE_NUM']
        cins = get_channel(cfg['CIN_MIN'], cfg['CIN_MAX'], cfg['CSTEP_MIN'])
        couts = get_channel(cfg['COUT_MIN'], cfg['COUT_MAX'], cfg['CSTEP_MIN'])
    
        while len(self.samples) < snum:
            dicts = {}
            cin = sampling(cins)
            dicts['CIN'] = cin 
            dicts['COUT'] = cin
            dicts = sample_other_params(cfg, dicts)
            self.samples.append(dicts)
            if dw  ==  False and cin * 2 <=  cfg['COUT_MAX']:

                dicts = {}
                cin = sampling(cins )
                dicts['CIN'] = cin 
                dicts['COUT'] = cin*2
                dicts = sample_other_params(cfg, dicts)
                self.samples.append(dicts)

           


class RandomSampler:
    def __init__(self, cfg, snum):
        self.cfg = cfg 
       
        self.samples = []
        #snum = cfg['SAMPLE_NUM']
        cins = get_channel(cfg['CIN_MIN'], cfg['CIN_MAX'], cfg['CSTEP_MIN'])
        couts = get_channel(cfg['COUT_MIN'], cfg['COUT_MAX'], cfg['CSTEP_MIN'])
    
        while len(self.samples)<snum:
            dicts = {}
            cin = sampling(cins )
            cout = sampling(couts)
            dicts['CIN'] = cin 
            dicts['COUT'] = cout 
            dicts = sample_other_params(cfg, dicts)

            
            self.samples.append(dicts)
            print(dicts)
        #print(self.samples)


class Sampling:
    def __init__(self, cfg):
        self.cfg = cfg 

        self.samples = []
        snum = cfg['SAMPLE_NUM']
        block_type = cfg['BLOCK_TYPE']
        if 'dwconv' not in block_type:
            basic_num = int(0.6*snum)
            r_num = snum-basic_num
            self.samples.extend(BasePatternSampler(cfg, basic_num).samples)
            self.samples.extend(RandomSampler(cfg, r_num).samples)
        else:
            self.samples.extend(BasePatternSampler(cfg, snum, dw = True).samples)
