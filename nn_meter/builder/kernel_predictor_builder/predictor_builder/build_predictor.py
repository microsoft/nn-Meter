# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
from sklearn.model_selection import train_test_split

from .predictor_zoo import init_predictor
from .extract_features import get_features_by_config
from .utils import latency_metrics, get_config_by_features


def build_predictor_by_data(kernel_type, data, hardware = 'cpu', error_threshold = 0.2):
    """
    build regression model by sampled data and latency, locate data with large-errors
    
    @params
    hardware (str): target device, relative to 

    result_save_path: folder that stores the measured csv files

    kernel: str 
    identical kernel name 
    error_threshold: float, default = 0.2, should be no less than 0.1 
    if prediction error >this threshold, we treat this data as a large-error-data. 
    ----------
    Returns
    10% Accuracy: float 
    cfgs: configuration list, where each item is a configuration for the large-error-data
    """
    

    # get data for regression
    X, Y = get_features_by_config(kernel_type, config=data)
    
    # initialize the regression model based on `RandomForestRegressor`
    predictor = init_predictor(kernel_type, hardware)
    
    # start training
    trainx, testx, trainy, testy = train_test_split(X, Y, test_size = 0.2, random_state = 10)
    predictor.fit(trainx, trainy)
    predicts = predictor.predict(testx)
    rmse, rmspe, error, acc5, acc10, acc15 = latency_metrics(predicts, testy)
    logging.info(f"rmse: {rmse}; rmspe: {rmspe}; error: {error}; 5% accuracy: {acc5}; 10% accuracy: {acc10}; 15% accuracy: {acc15}.")

    # locate large error data
    error_configs = []
    for i in range(len(testx)):
        y1, y2 = testy[i], predicts[i]
        error = abs(y1 - y2) / y1
        if error > error_threshold:
            error_config = get_config_by_features(kernel_type, testx[i])
            error_configs.append(error_config)
    
    return predictor, acc10, error_configs
