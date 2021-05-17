import sys,os,pickle
sys.path.append("..")
from predictors.extract_feature import*
def get_singleop_feature(features,opname,hardware):
    nfs=[]
    #print(features)
    for fe in features:
       # print(fe)
        if len(fe)>4:
            [inputh,cin,cout,ks,s,flops,params]=fe
            if s==2:
                nfe=[inputh/2,cout]
            else:
                nfe=[inputh,cout]
        elif len(fe)==4:
            [cin,cout,flops,params]=fe 
            nfe=[1,cout]
        elif len(fe)==2:
            #print(fe)
            [hw,cin]=fe 
            nfe=[hw,cin]
            #print('here1',opname)
            if opname=='add':
                [hw,cin]=fe 
                nfe=[hw,cin,cin]
               # print('here')
        elif len(fe)==3:
            [hw,cin1,cin2]=fe 
            nfe= [hw,cin1,cin2]
            #print('opname',opname)
            if opname=='relu':
                [hw,cin1,cin2]=fe 
                nfe=[hw,cin1]
        else:
           # print('here4')
            [hw,cin1,cin2]=fe 
            nfe=[hw,cin1]
        nfs.append(nfe)
    return nfs

def get_op_predictor_name(optype,hardware):
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
    if optype=='global-pool':
        optype='global-avgpool'
    if optype=='channel_shuffle':
        optype='channelshuffle'
    if optype in ['bn-relu']:
        optype='bnrelu'
    if optype in ['add-relu','add']:
        optype='addrelu'
    if hardware=='cpu' and optype =='add':
        optype='addrelu'
    if hardware=='gpu' and optype=='addrelu':
        return 'relu'
   
        
    if optype in ['SE','SE-relu']:
        optype='se'
    return optype
    
def get_op_latency(optype,features,hardware,path="data/predictorzoo"):
    latency=0 
    optype=get_op_predictor_name(optype,hardware)
    modelname=path+'/'+hardware+'/'+optype+'.pkl'
   # print(modelname)
    if os.path.isfile(modelname):
        if ( 'fc' in modelname or 'global' in modelname  ) and hardware=='gpu':
            return []
        with open(modelname,'rb') as f:
            model=pickle.load(f)
            predicts=model.predict(features)
            return predicts
    return []

def predict_model(model,hardware):
    py=0
    dicts={}
    for layer in model:
                
        op=list(model[layer].keys())[0]
        features=model[layer][op]
                #print(layer,op,features)
        if not op in dicts:
                dicts[op]=[]
        dicts[op].append(features)
    for op in dicts:
            #print(op,'name')
            opitems=op.split('-')
            if hardware=='gpu' and op in ['add','add-relu']:
                nfes=[]
                for fe in dicts[op]:
                    nfes.append(fe[0:2])
                pys=get_op_latency(op,nfes,hardware)
            else:
                pys=get_op_latency(op,dicts[op],hardware)    
            #print(op,sum(pys))  
           # print(opitems)
            if len(pys)!=0:
                py+=sum(pys)
            if 'hswish' in op and op!='hswish':
                    nfs=get_singleop_feature(dicts[op],'hswish',hardware)
                    hspys=get_op_latency('hswish',nfs,hardware)
                   # print('hswish')
                    if len(pys)!=0:
                        py+=sum(hspys)
            if 'bn' in op and op!='bn':
                   # print('here bn')
                    nfs=get_singleop_feature(dicts[op],'bn',hardware)
                   # print(nfs)
                    hspys=get_op_latency('bn',nfs,hardware)
                    
                    if len(pys)!=0:
                        bnsum=sum(hspys)
                        py+=opitems.count('bn')*bnsum
                       # print(hspys,bnsum)
                       # print('bn',opitems.count('bn')*bnsum,len(hspys),opitems.count('bn'))
                   # print(op,hspys)
            if 'relu' in op and op!='relu':
                    nfs=get_singleop_feature(dicts[op],'relu',hardware)
                   # print('here relu')
                    #print('relu',nfs)
                    hspys=get_op_latency('relu',nfs,hardware)
                    if len(pys)!=0:
                        relusum=sum(hspys)
                        py+=opitems.count('relu')*relusum
                       # print('relu',opitems.count('relu')*relusum)
                        op='relu'
                  # print(op,hspys)
                    #print('finish')
            if 'add' in op and op!='add':
                    #print('here3 add')
                    nfs=get_singleop_feature(dicts[op],'add',hardware)
                    #print(nfs)
                    hspys=get_op_latency('addrelu',nfs,hardware)
                    #print(hspys)
                    if len(pys)!=0:
                        addsum=sum(hspys)
                        py+=opitems.count('add')*addsum
                        op='add'
                        #print('relu',opitems.count('add')*addsum)
                  # print(op,hspys)
    return py
def main_op_predict(hardware,mf,configs,latencyfile):
    X,Y=read_model_latency(configs,latencyfile)
    pY=[]
    if os.path.isdir("results/op_predicts")==False:
        os.mkdir("results/op_predicts")
    f=open("results/op_predicts/result-"+hardware+'-'+mf+'-nofuse.csv','w')

    Y1=[]

    print(len(X))
    for index in X:
      
        #if index.endswith('0'):
        #if index in ['proxylessnas_1']:
           # print(index)
            model=X[index]
            latency=Y[index]
            py=predict_model(model,hardware)
            
            pY.append(py)
            Y1.append(Y[index])    
            error=abs(latency-py)/latency
            print(index,py,latency,error)
            f.write(index+','+str(py)+','+str(latency)+','+str(error)+'\n')
            #break
            
        
 
    rmse,rmspe,error,acc5,acc10,acc15=lat_metrics(np.array(pY),np.array(Y1))
    print(rmse,rmspe,error,acc5,acc10)
    

   

