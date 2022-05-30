import os
import json

import argparse
import tqdm

import subprocess
import multiprocessing

def build_tiny_net(config_dict):
    subprocess.check_output(f'python3 build_tiny_net.py -i "{config_dict["arch_str"]}" ' +
                            f'-o {config_dict["pb_file_name"]} ' + (f'-t {config_dict["tflite_file_name"]}' if config_dict["tflite_file_name"] !=  '' else '')
                            , shell = True, stderr = open(os.devnull, 'w'))

if __name__  ==  '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', 
        '--input_nasbench201_descriptor', 
        type = str, 
        required = True)
    parser.add_argument(
        '-o', 
        '--output_folder', 
        type = str, 
        required = True)
    parser.add_argument(
        '-f', 
        '--output_tflite_folder', 
        type = str, 
        default = '')
    parser.add_argument(
        '-t', 
        '--num_of_thread', 
        type = int, 
        default = 12)
    parser.add_argument(
        '-n', 
        '--num_of_samples', 
        type = int, 
        default = 2000)
    args = parser.parse_args()   

    nasbench201_descriptor = json.loads(open(args.input_nasbench201_descriptor, 'r').read())
    nasbench201_acc_seq = sorted(nasbench201_descriptor, key = lambda x: nasbench201_descriptor[x]['acc'], reverse = True)
    
    nasbench201_acc_seq = nasbench201_acc_seq[:args.num_of_samples]

    #Pack args
    build_tiny_net_args = []
    for nasbench_keys in nasbench201_acc_seq:
        arch_str = nasbench201_descriptor[nasbench_keys]['config']['arch_str']
        pb_file_name = os.path.abspath(os.path.join(args.output_folder, 'nasbench201_%s.pb' % nasbench_keys))
        if args.output_tflite_folder !=  '':
            tflite_file_name = os.path.abspath(os.path.join(args.output_tflite_folder, 'nasbench201_%s.tflite' % nasbench_keys))
        else:
            tflite_file_name = ''
        build_tiny_net_args.append({'arch_str': arch_str, 'pb_file_name': pb_file_name, 'tflite_file_name': tflite_file_name})

    if not os.path.isdir(args.output_folder):
        os.mkdir(args.output_folder)
    
    if args.output_tflite_folder !=  '':
        if not os.path.isdir(args.output_tflite_folder):
            os.mkdir(args.output_tflite_folder)

    with multiprocessing.Pool(processes = args.num_of_thread) as p: 
        for _ in tqdm.tqdm(p.imap_unordered(build_tiny_net, build_tiny_net_args), total = len(build_tiny_net_args)):
                pass


