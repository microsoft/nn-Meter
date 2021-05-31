# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from prediction.predictors.predict_by_kernel import nn_predict
from prediction.predictors.predict_by_kernel import main_kernel_predict
from kerneldetection import KernelDetector
from ir_converters import model_file_to_graph
from prediction.load_predictors import*
import argparse
from config import BACKENDS
import os


def main(hardware, model, rule_file):
    graph = model_file_to_graph(model)
    #print(graph)
    kd = KernelDetector(rule_file)
    kd.load_graph(graph)
    #print(model)
    mid=model.split('/')[-1].replace(".onnx","").replace(".pb","").replace(".json","")
    kernel_result={mid:kd.kernels}
    predictor=load_lat_predictors(hardware)
    py=nn_predict(predictor,kernel_result)
    print('predict the latency of '+mid+" on "+hardware+": "+str(py))
    '''
    mf=model.split('/')[-1].split('_')[0].replace("small","").replace("large","")
    mf=mf.replace("11","").replace("13","").replace("16","").replace("19","")
    mf=mf.replace("18","").replace("34","").replace("50","")
    latencyfile='data/model_latency/'+hardware+'/'+mf+"-log.csv"
    main_kernel_predict(predictor,kernel_result,latencyfile)
    '''




if __name__ == '__main__':
    parser = argparse.ArgumentParser('predict model latency on device')
    parser.add_argument('-hw', '--hardware', type=str, default='cpu')
    parser.add_argument('-i', '--input_model', type=str, required=True, help='Path to input model. ONNX, FrozenPB or JSON')
    parser.add_argument('-r', '--rule_file', type=str, help='Specify path to rule file. Default set by config.py and hardware.')

    args=parser.parse_args()
    
    rule_file = args.rule_file or BACKENDS[args.hardware]
        
    main(args.hardware, args.input_model, rule_file)
