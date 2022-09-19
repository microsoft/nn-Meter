# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import json
import requests
import numpy as np
from tqdm import tqdm
from zipfile import ZipFile


def download_from_url(urladdr, ppath):
    """
    download the kernel predictors from the url
    @params:

    urladdr: github release url address
    ppath: the targeting dir to save the download data (usually hardware_inferenceframework)

    """
    file_name = os.path.join(ppath, ".zip")
    if not os.path.isdir(ppath):
        os.makedirs(ppath)

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
    zipfile.close() 
    progress_bar.close()
    os.remove(file_name)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (bytes, bytearray)):
            return obj.decode("utf-8")
        return json.JSONEncoder.default(self, obj)
