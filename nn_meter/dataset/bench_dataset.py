# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import logging
import jsonlines
from glob import glob
from nn_meter.predictor import latency_metrics, list_latency_predictors, load_latency_predictor
from nn_meter.utils import download_from_url, get_user_data_folder
logging = logging.getLogger("nn-Meter")


__user_dataset_folder__ = os.path.join(get_user_data_folder(), 'dataset')

def bench_dataset(url="https://github.com/microsoft/nn-Meter/releases/download/v1.0-data/datasets.zip",
                  data_folder=__user_dataset_folder__):
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
        logging.keyinfo(f'Download from {url} ...')
        download_from_url(url, data_folder)

    datasets = glob(os.path.join(data_folder, "**.jsonl"))
    return datasets
        
if __name__ == '__main__':

    datasets = bench_dataset()
    hws = list_latency_predictors()
    
    for hw in hws:
        hw_name, hw_version = hw["name"], hw["version"]
        predictor = load_latency_predictor(hw_name, hw_version)
        for filename in datasets:
            True_lat = []
            Pred_lat = []
            index = 0
            with jsonlines.open(filename) as data_reader:
                for i, item in enumerate(data_reader):
                    graph = item["graph"]
                    pred_lat = predictor.predict(graph, model_type="nnmeter-ir")
                    real_lat = item[hw_name]
                    logging.result(f'{filename}[{i}]: predict: {pred_lat}, real: {real_lat}')
                    if real_lat != None:
                        True_lat.append(real_lat)
                        Pred_lat.append(pred_lat)
                        index += 1
            if len(True_lat) > 0:
                rmse, rmspe, error, acc5, acc10, _ = latency_metrics(Pred_lat, True_lat)
                logging.result(
                    f'{filename} on {hw_name}: rmse: {rmse}, 5%accuracy: {acc5}, 10%accuracy: {acc10}'
                )
