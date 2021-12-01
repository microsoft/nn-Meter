from .prior_distribution_sampler import*
from .finegrained_sampler import*
def add_to_cfg(hws,ks,strides,cins,couts=None):
    cfgs=[]
    for index in range(len(hws)):
        dicts={}
        dicts['HW']=hws[index]
        dicts['CIN']=cins[index]
        dicts['KERNEL_SIZE']=ks[index]
        dicts['STRIDE']=strides[index]
        if couts:
            dicts['COUT']=couts[index]
        cfgs.append(dicts)
    return cfgs

def add_to_cfg_fc(cins,couts):
    cfgs=[]
    for index in range(len(cins)):
        dicts={}
       
        dicts['CIN']=cins[index]
        
        dicts['COUT']=couts[index]
       
        cfgs.append(dicts)
    return cfgs
def add_to_cfg_gp(hws,cins):
    cfgs=[]
    for index in range(len(cins)):
        dicts={}
       
        dicts['CIN']=cins[index]
        dicts['HW']=hws[index]
        
       
        cfgs.append(dicts)
    return cfgs
def add_to_cfg_addrelu(hws,cins,cins1):
    cfgs=[]
    for index in range(len(cins)):
        dicts={}
       
        dicts['CIN']=cins[index]
        dicts['COUT']=cins1[index]
        dicts['HW']=hws[index]
        
       
        cfgs.append(dicts)
    return cfgs

def add_to_cfg_concat(hws,ns,cin1,cin2,cin3,cin4):
    cfgs=[]
    cins=[cin1,cin2,cin3,cin4]
    for index in range(len(hws)):
        dicts={}
        dicts['HW']=hws[index]
        n=ns[index]
        if n==2:
            c=[cin1[index],cin2[index]]
        if n==3:
            c=[cin1[index],cin2[index],cin3[index]]
        if n==4:
            c=[cin1[index],cin2[index],cin3[index],cin4[index]]
        dicts['N']=n 
        dicts['CINS']=c 
        cfgs.append(dicts)
    return cfgs
    


def block_sampling_with_finegrained(blocktype,count,cfgs):
        """
        finegrained sampling
        Parameters
        ----------
        blocktype: str
        identical kernel name 
        count: int 
        for each large-error-data-point, we sample `count` more data around it.
        cfgs: list
        each item in the list represent a large-error-data-point. each item is a dictionary, storing the configuration
        -----------
        Return
        list, sampled data configurations
        
        """
  
        if blocktype in['conv-bn-relu','conv-bn-relu6','conv-bn','conv-relu','conv-relu6','conv-bn-hswish','conv-hswish']:
            return finegrained_sampling_conv(cfgs,count)
        if blocktype in['dwconv-bn-relu','dwconv-bn-relu6','dwconv-bn','dwconv-relu','dwconv-relu6','dwconv-bn-hswish','maxpool','avgpool']:
            return finegrained_sampling_dwconv(cfgs,count)
        if blocktype in ['fc']:
            return finegrained_sampling_fc(cfgs,count)
       
        if blocktype in ['bnrelu','bn','relu','global-avgpool']:
            return finegrained_sampling_CIN(cfgs,count)
        if blocktype in ['split','hswish','se','channel-shuffle']:
            return finegrained_sampling_CIN1(cfgs,count)
        if blocktype in ['addrelu','add']:
            return finegrained_sampling_CIN(cfgs,count)
           # print(cfgs)
        if blocktype in ['concat']:
            return finegrained_sampling_concat(cfgs,count)
        

def block_sampling_from_prior(blocktype,count):
        """
        initial sampling
        Parameters
        ----------
        blocktype: str
        identical kernel name 
        count: int 
        for the target kernel, we sample `count` data from prior distribution.

        -----------
        Return
        list, sampled data configurations
        
        """
        if blocktype in['conv-bn-relu','conv-bn-relu6','conv-bn','conv-relu','conv-relu6','conv-bn-hswish','conv-hswish']:
         
            cins,couts,hws,ks,strides=sampling_conv(count)
            return add_to_cfg(hws,ks,strides,cins,couts)
        
        
        if blocktype in['dwconv-bn-relu','dwconv-bn-relu6','dwconv-bn','dwconv-relu','dwconv-relu6','se','dwconv-bn-hswish']:
            cs,hws,ks,strides=sampling_dwconv(count)
            return add_to_cfg(hws,ks,strides,cs)
        if blocktype in ['fc']:
          
            cins,couts=sampling_fc(int(count*0.5),fix_cout=1000)
            cins1,couts1=sampling_fc(int(count*0.5),fix_cout=False)
            cins.extend(cins1)
            couts.extend(couts1)
            return add_to_cfg_fc(cins,couts)
        if blocktype in ['maxpool']:
           
            hws,cins,ks,strides=sampling_pooling(count)
         

            return add_to_cfg(hws,ks,strides,cins)

        if blocktype in ['avgpool']:
           
            hws,cins,ks,strides=sampling_pooling(count)
            ks=[3]*len(ks)
            strides=[1]*len(ks)

            return add_to_cfg(hws,ks,strides,cins)
        if blocktype=='global-avgpool':

            cins,couts=sampling_fc(count)
            hws=[7]*count 
            return add_to_cfg_gp(hws,cins)
        if blocktype in ['split','channel_shuffle','hswish','bnrelu','bn','relu']:
            cs,hws,ks,strides=sampling_dwconv(count,resize=True)
            ncs=[]
            for c in cs:
                nc=c 
                if c%2!=0:
                    nc=c+1 
                ncs.append(nc)
            return add_to_cfg_gp(hws,ncs)
        if blocktype=='concat':
            hws,ns,cin1,cin2,cin3,cin4=sampling_concats(count)
            return add_to_cfg_concat(hws,ns,cin1,cin2,cin3,cin4)
        if blocktype=='addrelu':
            hws,cs1,cs2=sampling_addrelu(count)
            return add_to_cfg_addrelu(hws,cs1,cs2)





         
