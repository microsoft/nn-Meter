# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import sys
sys.path.append("prediction")
from predictors.utils import*
from predictors.extract_feature import*

def merge_op(op):
    if 'conv' in op and 'dwconv' not in op:
        return 'conv-bn-relu'
    elif 'dwconv' in op:
        return 'dwconv-bn-relu'
    else:
        return op
    
    

def predict_model(model,predictors):
    py=0
    dicts={}
    for layer in model:
            op=list(model[layer].keys())[0]
            features=model[layer][op]
            rop=merge_op(op)
            if not rop in dicts:
                dicts[rop]=[]
            dicts[rop].append(features)
         
    for op in dicts:
          #  print(op)
            opname=get_kernel_name(op)
            if opname in predictors:
                pred=predictors[opname]
                pys=pred.predict(dicts[op])
                #pys=get_kernel_latency(op,dicts[op],hardware)  
                if len(pys)!=0:
                    py+=sum(pys)
            
            

    return py

def nn_predict(predictor,configs):
    if isinstance(configs,dict):
       # print(configs.items())
        config=configs[list(configs.keys())[0]]
    else:
        config=configs 
    features=get_predict_features(config)
    py=predict_model(features,predictor)
    print(py)
    return py

def main_kernel_predict(predictor,configs,latencyfile):
    Y=read_model_latency(latencyfile)
    pY=[]#
    Y1=[]
    for mid in configs:
        features=get_predict_features(configs[mid])
        py=predict_model(features,predictor)
        if mid in Y:
            y=Y[mid]
            pY.append(py)
            Y1.append(y)
            print(mid,'predict '+str(py)+',real '+str(y))

    rmse,rmspe,error,acc5,acc10,acc15=lat_metrics(np.array(pY),np.array(Y1))
    print('rmse',rmse,'rmpse',rmspe,'error',error,'acc',acc10)
    return rmse, rmspe,error, acc5,acc10
