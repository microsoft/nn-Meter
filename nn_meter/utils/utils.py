# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
from zipfile import ZipFile
from tqdm import tqdm
import requests


def download_from_url(urladdr, ppath):
    """
    download the kernel predictors from the url
    @params:

    urladdr: github release url address
    ppath: the targeting hardware_inferenceframework name

    """
    file_name = ppath + "/" + ".zip"
    if not os.path.isdir(ppath):
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
