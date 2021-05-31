import pickle,os
from glob import glob
def load_lat_predictors(hardware,dir="data/predictorzoo"):
    ppath=dir+"/"+hardware
    #print(ppath)
    if os.path.isdir(ppath):
        predictors={}
        ps=glob(ppath+"/**.pkl")
       # print(ps)
        for p in ps:
            
            pname=p.split('/')[-1].replace(".pkl","")
            with open(p,'rb') as f:
                print('load predictor',p)
                model=pickle.load(f)            
                predictors[pname]=model
        return predictors

    else:
        raise Exception('nn-Meter currently does not support '+hardware)

