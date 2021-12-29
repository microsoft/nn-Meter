# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
from .extract_feature import get_feature
from .utils import lat_metrics
from .predictor_zoo import get_model
from sklearn.model_selection import train_test_split
from math import sqrt


def get_config_by_features(kernelname, feature):
    cfg = {}
    pass
    return cfg 


def build_predictor(block_type, hardware, dir, error_threshold = 0.2):
    """
    build regression model by sampled data and latency, locate data with large-errors
    Parameters
    ----------
    hardware: target device 
    result_save_path: folder that stores the measured csv files
    kernel: str 
    identical kernel name 
    error_threshold: float, default = 0.2, should be no less than 0.1 
    if prediction error >this threshold, we treat this data as a large-error-data. 
    ----------
    Returns
    10% accuracy: float 
    cfgs: configuration list, where each item is a configuration for the large-error-data
    """
    cfgs = []
    # get data for regression
    X, Y = ...
    
    # initialize the regression model based on `RandomForestRegressor`
    model = get_model(block_type, hardware)
    
    # start training
    trainx, testx, trainy, testy = train_test_split(
        X, Y, test_size = 0.2, random_state = 10
    )
    model.fit(trainx, trainy)
    predicts = model.predict(testx)
    # locate large error data
    for i in range(len(testx)):
        y1 = testy[i]
        y2 = predicts[i]
        error = abs(y1-y2)/y1
        if error > error_threshold:
            print(testx[i])
            cfg = get_config_by_features(block_type, testx[i])
            cfgs.append(cfg)
    rmse, rmspe, error, acc5, acc10, acc15 = lat_metrics(predicts, testy)
    print(f"rmse: {rmse}; rmspe: {rmspe}; error: {error}; 5% accuracy: {acc5}; 10% accuracy: {acc10}; 15% accuracy: {acc15}.")
    return acc10, cfgs
