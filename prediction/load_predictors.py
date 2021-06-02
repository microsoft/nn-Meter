import pickle,os
from glob import glob
import shutil
from zipfile import ZipFile
from tqdm import tqdm
import requests

import zipfile, requests, shutil
def load_lat_predictors(configs,dir="data/predictorzoo"):
    hardware=configs['name']
    ppath=dir+"/"+hardware
    #print(ppath)
    isdownloaded=check_predictors(ppath,configs['predictors_num'])
    if isdownloaded==False:  ## todo
        download_from_url(configs['download'],dir,hardware)
    

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

    
def download_from_url(urladdr,ppath,filename):
    file_name=ppath+"/"+'.zip'
    if os.path.isdir(ppath)==False:
        os.makedirs(ppath)
    '''
    r = requests.get(urladdr, stream=True)
    f = open(file_name, "wb")
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:
          f.write(chunk）
    '''
    print('download from '+urladdr)
    response = requests.get(urladdr, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 2048 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(file_name, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    zipfile = ZipFile(file_name)
    zipfile.extractall(path=ppath)
    progress_bar.close()
    os.remove(file_name)
    #http_response = urlopen(urladdr)
    #total_length = int(http_response.headers.get('content-length'))
    #print('here1',total_length)
    
  
   
   


def check_predictors(ppath,predictor_num):

    if os.path.isdir(ppath):
        ps=glob(ppath+"/**.pkl")
        if len(ps)!=predictor_num:
            return False
        else:
            return True
    else:
        return False
