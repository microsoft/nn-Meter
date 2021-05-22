from prediction.predictors.predict_by_kernel import main_kernel_predict
from kerneldetection import KernelDetector
from ir_converters import model_file_to_grapher
import argparse
from config import BACKENDS
import os


def main(hardware, model, rule_file, mf, level, latency_file):
    graph = model_file_to_grapher(model)
    kd = KernelDetector(rule_file)
    kd.load_graph(graph)
    print(model)
    mid=model.split('/')[-1].replace(".onnx","").replace(".pb","")
    kernel_result={mid:kd.kernels}
    #print(kernel_result)

    if level == 'kernel':
        rmse, rmspe, error, acc5, acc10 = main_kernel_predict(hardware, mf, kernel_result, latency_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('predict model latency on device')
    parser.add_argument('-hw', '--hardware', type=str, default='cpu')
    parser.add_argument('-m', '--mf', type=str, default='alexnet')
    parser.add_argument('-l', '--level', type=str, default='kernel')
    parser.add_argument('-i', '--input_model', type=str, required=True, help='Path to input model. ONNX, FrozenPB or JSON')
    parser.add_argument('-r', '--rule_file', type=str, help='Specify path to rule file. Default set by config.py and hardware.')
    parser.add_argument('-lf', '--latency_file', type=str)

    args=parser.parse_args()

    rule_file = args.rule_file or BACKENDS[args.hardware]
    latency_file = args.latency_file or f'data/model_latency/{args.hardware}/{args.mf}-log.csv'
   
    main(args.hardware, args.input_model, rule_file, args.mf, args.level, latency_file)
