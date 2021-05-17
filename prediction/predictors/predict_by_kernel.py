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
    
    

def predict_model(model,hardware):
    py=0
    dicts={}
    for layer in model:
            op=list(model[layer].keys())[0]
            features=model[layer][op]
           # print(op,features)
            rop=merge_op(op)
            if not rop in dicts:
                dicts[rop]=[]
            dicts[rop].append(features)
         
    for op in dicts:
           # print('op',op)
            pys=get_kernel_latency(op,dicts[op],hardware)
            
            if len(pys)!=0:
                py+=sum(pys)
                #if 'conv' in op:
                #     print(op,len(pys),sum(pys),dicts[op],pys)
                #print(op,len(pys),pys,sum(pys),dicts[op])
                #print(dicts[op])
                #print('curent',py)
            #if op!='hswish' and 'hswish' in op:
             #   nfs=get_hswish_feature(dicts[op])
              #  hspys=get_kernel_latency('hswish',nfs,hardware)
             #   if len(pys)!=0:
             #       py+=sum(hspys)
            
            

    return py
def main_kernel_predict(hardware,mf,configs,latencyfile):
    
   
       
    X,Y=read_model_latency(configs,latencyfile)
    pY=[]#
    f=open("results/result-"+hardware+'-'+mf+'.csv','w')
    Y1=[]
    #print(len(X))
    count=0
    for index in X:
       #if index in ['mobilenetv1_0']:
       #if 'small' in index:
            model=X[index]
            #print(model)
            latency=Y[index]
            py=predict_model(model,hardware)                
            pY.append(py)
            Y1.append(Y[index])
            error=abs(latency-py)/latency
            print(index,py,latency,error)            
            f.write(str(index)+','+str(py)+','+str(latency)+','+str(error)+'\n')
            rmse,rmspe,error,acc5,acc10,acc15=lat_metrics(np.array(pY),np.array(Y1))
            count+=1
            #break



    rmse,rmspe,error,acc5,acc10,acc15=lat_metrics(np.array(pY),np.array(Y1))
    print('rmse',rmse,'rmpse',rmspe,'error',error,'acc',acc10)
    return rmse, rmspe,error, acc5,acc10
