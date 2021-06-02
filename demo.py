# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from nn_meter import load_latency_predictors
import yaml
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser('predict model latency on device')
    parser.add_argument('-i', '--input_model', type=str, required=True, help='Path to input model. ONNX, FrozenPB or JSON')
    parser.add_argument('-c', '--config', type=str, required=True, help='config file')

    args=parser.parse_args()

    with open(args.config) as file:
        config=yaml.load(file,Loader=yaml.FullLoader)
        print(config)        
        predictor=load_latency_predictors(config)
        latency=predictor.predict(args.input_model)
    
