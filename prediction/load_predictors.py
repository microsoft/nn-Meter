import pickle,os
from glob import glob
import zipfile, requests, shutil
def load_lat_predictors(configs,dir="data/predictorzoo"):
    hardware=configs['name']
    ppath=dir+"/"+hardware
    #print(ppath)
    isdownloaded=check_predictors(ppath,configs['predictors_num'])
    if isdownloaded==False:  ## todo
        download_from_url(configs['download'],ppath)
    

    ##load predictors
    predictors={}
    ps=glob(ppath+"/**.pkl")

    for p in ps:
            
            pname=p.split('/')[-1].replace(".pkl","")
            with open(p,'rb') as f:
                print('load predictor',p)
                model=pickle.load(f)            
                predictors[pname]=model
    fusionrule=ppath+'/rule_'+hardware+'.json'
    print(fusionrule)
    if os.path.isfile(fusionrule)==False:
        raise ValueError("check your fusion rule path, file "+fusionrule+' does not exist！')
    return predictors,fusionrule

    
def download_from_url(urladdr,ppath):
    file_name=urladdr.split('/')[-1]
    if os.path.isdir(ppath)==False:
        os.makedirs(ppath)
    r = requests.get(urladdr, stream=True)
    f = open("file_path.zip", "wb")

    for chunk in r.iter_content(chunk_size=512):
        print('here')
        if chunk:
          f.write(chunk)


def check_predictors(ppath,predictor_num):
    if os.path.isdir(ppath):
        ps=glob(ppath+"/**.pkl")
        if len(ps)!=predictor_num:
            return False
        else:
            return True
    else:
        return False
