import pickle, os
from glob import glob
import shutil
from zipfile import ZipFile
from tqdm import tqdm
import requests


def loading_to_local(configs, hardware, dir="data/predictorzoo"):
    if hardware not in configs:
        raise NotImplementedError
    ppath = dir + "/" + hardware
    isdownloaded = check_predictors(ppath, configs[hardware]["kernel_predictors"])
    if isdownloaded == False:  ##
        download_from_url(configs[hardware]["download"], dir, hardware)

    ##load predictors
    predictors = {}
    ps = glob(ppath + "/**.pkl")
    for p in ps:
        pname = p.split("/")[-1].replace(".pkl", "")
        with open(p, "rb") as f:
            print("load predictor", p)
            model = pickle.load(f)
            predictors[pname] = model
    fusionrule = ppath + "/rule_" + hardware + ".json"
    print(fusionrule)
    if os.path.isfile(fusionrule) == False:
        raise ValueError(
            "check your fusion rule path, file " + fusionrule + " does not exist！"
        )
    return predictors, fusionrule


def download_from_url(urladdr, ppath, filename):
    file_name = ppath + "/" + ".zip"
    if os.path.isdir(ppath) == False:
        os.makedirs(ppath)

    print("download from " + urladdr)
    response = requests.get(urladdr, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 2048  # 2 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(file_name, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    zipfile = ZipFile(file_name)
    zipfile.extractall(path=ppath)
    progress_bar.close()
    os.remove(file_name)


def check_predictors(ppath, kernel_predictors):
    print('checking local kernel predictors at '+ppath)
    if os.path.isdir(ppath):
        filenames = glob(ppath + "/**.pkl")  ## 
        ## check if all the pkl files are included
        for kp in kernel_predictors:
            fullpath=ppath + "/" + kp + ".pkl"
            if fullpath not in filenames:
                return False 
        return True       
    else:
        return False
