# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
from sklearn.model_selection import train_test_split
from .utils import latency_metrics
from .predictor_lib import init_predictor
from .extract_features import get_feature_parser, get_data_by_profiled_results


def build_predictor_by_data(kernel_type, kernel_data, backend = None, error_threshold = 0.1):
    """
    build regression model by sampled data and latency, locate data with large-errors. Returns (current predictor, 10% Accuracy, error_cfgs), 
    where error_cfgs represent configuration list, where each item is a configuration for one large-error-data.

    @params
    kernel_type (str): type of target kernel

    data (tuple): feature (configs) and target (latencies) data

    backend (str): target device, relative to predictor initialization
    
    error_threshold (float): default = 0.1, should be no less than 0.1. if prediction error (`abs(pred - true) / true`) > error_threshold,
        we treat this data as a large-error-data.    
    """
    feature_parser = get_feature_parser(kernel_type)
    data = get_data_by_profiled_results(kernel_type, feature_parser, kernel_data)

    # get data for regression
    X, Y = data
    trainx, testx, trainy, testy = train_test_split(X, Y, test_size = 0.2, random_state = 10)
    logging.info(f"training data size: {len(trainx)}, test data size: {len(testx)}")
    
    # initialize the regression model based on `RandomForestRegressor`
    predictor = init_predictor(kernel_type, backend)
    
    # start training
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
            error_config = feature_parser.get_config_by_feature(testx[i])
            error_configs.append(error_config)
    
    return predictor, acc10, error_configs
