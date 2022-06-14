# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from .utils import collect_kernel_data, latency_metrics
from .predictor_lib import init_predictor
from .extract_feature import get_feature_parser, get_data_by_profiled_results
logging = logging.getLogger("nn-Meter")


def build_predictor_by_data(kernel_type, kernel_data, backend = None, error_threshold = 0.1, mark = '', save_path = None, predict_label = "latency", final_predictor=False):
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

    save_path (str): the folder to save results file such as feature table and predictor pkl file. If save_path is None, the data will not be saved.
    
    predict_label (str): the predicting label to build kernel predictor
    """
    feature_parser = get_feature_parser(kernel_type)
    if save_path:
        os.makedirs(os.path.join(save_path, "collection"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "predictors"), exist_ok=True)
        data_save_path = os.path.join(save_path, "collection", f'Data_{kernel_type}_{mark}.csv')
        res_save_path = os.path.join(save_path, "collection", f"TestResult_{kernel_type}_{mark}.csv")
        pred_save_path = os.path.join(save_path, "predictors", f"{kernel_type}_{mark}.pkl")
    else:
        data_save_path = None
        res_save_path = None
        pred_save_path = None

    kernel_data = collect_kernel_data(kernel_data, predict_label)
    data = get_data_by_profiled_results(kernel_type, feature_parser, kernel_data,
                                        save_path=data_save_path,
                                        predict_label=predict_label)

    acc10, error_configs = None, None
    X, Y = data
    # initialize the regression model based on `RandomForestRegressor`
    predictor = init_predictor(kernel_type, backend)

    if final_predictor:
        predictor.fit(X, Y)
    else:
        # get data for regression
        trainx, testx, trainy, testy = train_test_split(X, Y, test_size = 0.2, random_state = 10)
        logging.info(f"training data size: {len(trainx)}, test data size: {len(testx)}")

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
        if res_save_path:
            test_res.to_csv(res_save_path, index=False)
            logging.info(f"All test data and predicted results are stored in path {res_save_path}")

        # locate large error data
        error_configs = []
        for i in range(len(testx)):
            if pred_error_list[i] > error_threshold:
                error_config = feature_parser.get_config_by_feature(testx[i])
                error_configs.append(error_config)

    # dump the predictor model
    if pred_save_path:
        import pickle
        with open(pred_save_path, 'wb') as fp:
            pickle.dump(predictor, fp)
        logging.keyinfo(f"Saved the predictor for {kernel_type} in path {pred_save_path}.")

    return predictor, acc10, error_configs
