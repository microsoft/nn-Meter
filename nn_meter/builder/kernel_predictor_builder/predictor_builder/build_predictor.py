# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from .utils import latency_metrics
from .predictor_lib import init_predictor
from .extract_features import get_feature_parser, get_data_by_profiled_results
logging = logging.getLogger("nn-Meter")


def build_predictor_by_data(kernel_type, kernel_data, backend = None, error_threshold = 0.1, mark = '', save_path = None):
    """
    build regression model by sampled data and latency, locate data with large-errors. Returns (current predictor, 10% Accuracy, error_cfgs), 
    where error_cfgs represent configuration list, where each item is a configuration for one large-error-data.

    @params
    kernel_type (str): type of target kernel

    data (tuple): feature (configs) and target (latencies) data

    backend (str): target device, relative to predictor initialization
    
    error_threshold (float): default = 0.1, should be no less than 0.1. if prediction error (`abs(pred - true) / true`) > error_threshold,
        we treat this data as a large-error-data.

    mark (str): the mark for the running results. Defaults to ''.

    save_path (str): the folder to save results file such as feature table and predictor pkl file
    """
    feature_parser = get_feature_parser(kernel_type)
    data = get_data_by_profiled_results(kernel_type, feature_parser, kernel_data, save_path=os.path.join(save_path, f'Data_{kernel_type}_{mark}.csv'))

    # get data for regression
    X, Y = data
    trainx, testx, trainy, testy = train_test_split(X, Y, test_size = 0.2, random_state = 10)
    logging.info(f"training data size: {len(trainx)}, test data size: {len(testx)}")
    
    # initialize the regression model based on `RandomForestRegressor`
    predictor = init_predictor(kernel_type, backend)
    
    # start training
    predictor.fit(trainx, trainy)
    predicts = predictor.predict(testx)
    pred_error_list = [abs(y1 - y2) / y1 for y1, y2 in zip(testy, predicts)]
    rmse, rmspe, error, acc5, acc10, acc15 = latency_metrics(predicts, testy)
    logging.info(f"rmse: {rmse:.4f}; rmspe: {rmspe:.4f}; error: {error:.4f}; 5% accuracy: {acc5:.4f}; 10% accuracy: {acc10:.4f}; 15% accuracy: {acc15:.4f}.")
    
    # dump the test set with predicts to csv file
    test_res = pd.DataFrame(testx, columns=[f'feature{i}' for i in range(len(testx[0]))])
    test_res["True"] = testy
    test_res["Pred"] = predicts
    test_res["Error"] = pred_error_list
    test_res.to_csv(os.path.join(save_path, f"TestResult_{kernel_type}_{mark}.csv"), index=False)
    logging.info(f"All test data and predicted results are stored in path {os.path.join(save_path, f'TestResult_{kernel_type}_{mark}.csv')}")
    
    # dump the predictor model
    import pickle
    save_path = os.path.join(save_path, f"Predictor_{kernel_type}_{mark}.pkl")
    with open(save_path, 'wb') as fp:
        pickle.dump(predictor, fp)
    logging.keyinfo(f"Saved the predictor for {kernel_type} in path {save_path}.")

    # locate large error data
    error_configs = []
    for i in range(len(testx)):
        if pred_error_list[i] > error_threshold:
            error_config = feature_parser.get_config_by_feature(testx[i])
            error_configs.append(error_config)
    
    return predictor, acc10, error_configs
