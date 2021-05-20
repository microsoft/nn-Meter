from kerneldetection import KernelDetector
from ir_converters import model_file_to_grapher
import argparse
import json


BACKENDS = {
    'cpu': 'tflite_cpu',
    'gpu': 'tflite_gpu',
    'vpu': 'vpu',
}


def main(input_model, rule_file, output_path):
    graph = model_file_to_grapher(input_model)
    kd = KernelDetector(rule_file)
    kd.load_graph(graph)

    with open(output_path, 'w') as fp:
        json.dump(kd.kernels, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware', type=str, default='cpu')
    parser.add_argument('-i', '--input_model', type=str, required=True, help='Path to input models. Either pb or onnx')
    parser.add_argument('-o', '--output_path', type=str,  default='out.json')
    parser.add_argument('-r', '--rule_file', type=str,  default='data/fusionrules/rule_tflite_cpu.json')
    #parser.add_argument('-t', '--input_type', type=str, choices=['multi-m','single-m'], default='multi-m', help='input file type: multi-m or single-m')
    #parser.add_argument('-backend', '--backend', type=str, choices=['tflite_cpu','tflite_gpu','vpu'], default='tflite_cpu', help='Default preserve the original layer names. Readable will assign new kernel names according to types of the layers.')
    args = parser.parse_args()

    main(args.input_model, args.rule_file, args.output_path)
