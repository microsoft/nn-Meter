from ir_converters import model_file_to_grapher
import argparse
import json
import sys


def converter(input_model, output_path):
    result = model_file_to_grapher(input_model)

    if output_path:
        with open(output_path, 'w') as fp:
            json.dump(result, fp, indent=4)
    else:
        json.dump(result, sys.stdout, indent=4)
    return result


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_model', type=str, required=True)
parser.add_argument('-o', '--output_path', type=str, required=False)
args = parser.parse_args()

if __name__ == '__main__':
    converter(args.input_model, args.output_path)
