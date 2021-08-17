# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json, os, argparse
import numpy as np
from nn_meter.prediction import latency_metrics
from glob import glob
from nn_meter.utils.graph_tool import NumpyEncoder
from nn_meter.ir_converters import model_file_to_graph, model_to_graph
from nn_meter.nn_meter import load_latency_predictor
from nn_meter import download_from_url
import jsonlines

ppath = "dataset"
url = "https://github.com/microsoft/nn-Meter/releases/download/v1.0-data/datasets.zip"
download_from_url(url, ppath)
datasets = glob(os.path.join(ppath, "**.jsonl"))
print(datasets)
hws = [
    "cortexA76cpu_tflite21",
    "adreno640gpu_tflite21",
    "adreno630gpu_tflite21",
    "myriadvpu_openvino2019r2",
]
for hw in hws:
    predictor = load_latency_predictor(hw)
    for filename in datasets:
        Y = []
        P = []
        index = 0
        with jsonlines.open(filename) as reader:
            for obj in reader:
                graph = obj["graph"]
                latency = predictor.predict(graph, model_type="nnmeter-ir")
                y = obj[hw]
                print(filename, "predict:", latency, "real:", y)
                if y != None:
                    Y.append(y)
                    P.append(latency)
                    index += 1
        if len(Y) > 0:
            rmse, rmspe, error, acc5, acc10, _ = latency_metrics(P, Y)
            print(
                filename, hw, "rmse:", rmse, "5%accuracy:", acc5, "10%accuracy:", acc10
            )
