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
def get_kernel_latency(optype,features,hardware,path="data/predictorzoo"):
    latency=0
    optype=get_kernel_name(optype,hardware) 
    modelname=path+'/'+hardware+'/'+optype+'.pkl'
    #print(modelname)
    if os.path.isfile(modelname):
       # print(modelname,len(features))
        #print(features)
        if ( 'fc' in modelname or 'global' in modelname  ) and hardware in ['gpu','gpu1']:
            return []
        #if ( 'bnrelu' in modelname   ) and args.hardware=='movidius':
         #   return []
        #print(modelname,features)
        with open(modelname,'rb') as f:
            model=pickle.load(f)
            predicts=model.predict(features)
            return predicts
    return []

def get_kernel_name(optype,hardware):
   
    #optype1=optype
    #print(hardware,optype)
    if optype in['conv-relu','conv-bn-relu6','conv-relu6','conv-bn-relu','conv-bn','conv-bn-hswish','conv-hswish','conv']:
        optype='conv-bn-relu' ## for movidius: conv-bn-relu1
    
    if 'conv' in optype and 'dwconv' not in optype:
        optype='conv-bn-relu'
    

    if optype in ['dwconv-bn-relu','dwconv-bn-relu6','dwconv-bn-hswish','dwconv-hswish','dwconv-bn']:
        optype='dwconv-bn-relu'##for movidius, dwconv-bn-relu1/2
    if 'dwconv' in optype:
        optype='dwconv-bn-relu'
    if optype=='fc-relu':
        optype='fc'
    if optype=='max-pool':
        optype='maxpool'
    if optype=='avg-pool':
        optype='avgpool'
    if optype=='global-pool':
        optype='global-avgpool'
    if optype=='channel_shuffle':
        optype='channelshuffle'
    if optype in ['bn-relu']:
        optype='bnrelu'
    if optype in ['add-relu']:
        optype='addrelu'
    if hardware=='cpu' and optype =='add':
        optype='addrelu'
    
       # print('here')
        
    if optype in ['SE','SE-relu']:
        optype='se'
   # print(hardware,optype)
    return optype
    

    