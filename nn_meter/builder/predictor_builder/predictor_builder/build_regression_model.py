# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
from .extract_feature import *
from .predictor_zoo import *
from sklearn.model_selection import train_test_split
from math import sqrt


def get_config_by_features(kernelname, feature):
    cfg = {}
    return cfg 


def build_predictor(hardware, dir, kernel, large_error_threshold = 0.2):
    
    """
    build regression model by sampled data and latency, locate data with large-errors
    Parameters
    ----------
    hardware: target device 
    dir: str
    folder that stores the measured csv files
    kernel: str 
    identical kernel name 
    large_error_threshold: float, default=0.2, should be no less than 0.1 
    if prediction error >this threshold, we treat this data as a large-error-data. 
    ----------
    Returns
    10% accuracy: float 
    cfgs: configuration list, where each item is a configuration for the large-error-data
    """
   
    filename = os.path.join(dir, hardware, kernel + '.csv')
    cfgs = []
    if os.path.isfile(filename):
        print("reading from file:", filename, ", the targeting kernel is", kernel)
        X, Y = read_kernel_latency(filename)## read the sampled data and latency, extract the kernel prediction features
        print("total numbers of data ", len(X))
        kernelname = kernel.replace('-', '')
        model = get_model(hardware, kernelname) ## get regression model
        if model != None:
            trainx, testx, trainy, testy = train_test_split(
                X, Y, test_size = 0.2, random_state = 10
            )
            model.fit(trainx, trainy)
            predicts = model.predict(testx)
            ### locate large error data
            for i in range(len(testx)):
                y1 = testy[i]
                y2 = predicts[i]
                error = abs(y1-y2)/y1
                if error > large_error_threshold:
                    print(testx[i])
                    cfg = get_config_by_features(kernelname, testx[i])
                    cfgs.append(cfg)
            rmse, rmspe, error, acc5, acc10, acc15 = lat_metrics(predicts, testy)
            print(f"rmse: {rmse}; rmspe: {rmspe}; error: {error}; 5% accuracy: {acc5}; 10% accuracy: {acc10}; 15% accuracy: {acc15}.")
            return acc10, cfgs
    return None, cfgs
