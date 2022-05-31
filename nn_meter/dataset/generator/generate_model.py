# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import yaml
import json
import random
import tensorflow as tf
from tensorflow import keras
import os

import argparse
import tqdm

import subprocess
import multiprocessing

from .networks.tf_network import *
from .networks.utils import save_to_models


MODELZOO = {
    'alexnet':lambda input_tensor, cfg, version, flag: AlexNet(input_tensor, cfg, version, flag), 
    'mobilenetv1':lambda input_tensor, cfg, version, flag: MobileNetV1(input_tensor, cfg, version, flag), 
    'mobilenetv2':lambda input_tensor, cfg, version, flag: MobileNetV2(input_tensor, cfg, version, flag), 
    'mobilenetv3':lambda input_tensor, cfg, version, flag: MobileNetV3(input_tensor, cfg, version, flag), 
    'shufflenetv2':lambda input_tensor, cfg, version, flag: ShuffleNetV2(input_tensor, cfg, version, flag), 
    'vgg':lambda input_tensor, cfg, version, flag: VGG(input_tensor, cfg, version, flag), # vgg11/ 13/ 16/ 19
    'resnet':lambda input_tensor, cfg, version, flag: ResNetV1(input_tensor, cfg, version, flag), 
    'proxylessnas':lambda input_tensor, cfg, version, flag: ProxylessNAS(input_tensor, cfg, version, flag), 
    'mnasnet':lambda input_tensor, cfg, version, flag: MnasNet(input_tensor, cfg, version, flag), 
    'squeezenet':lambda input_tensor, cfg, version, flag: SqueezeNet(input_tensor, cfg, version, flag), 
    'googlenet':lambda input_tensor, cfg, version, flag: GoogleNet(input_tensor, cfg, version, flag), 
    'densenet':lambda input_tensor, cfg, version, flag: DenseNet(input_tensor, cfg, version, flag), 
}


class generation:
    def __init__(self, config, savepath,):
        self.savepath = savepath
        self.log = {}
        with open(config, "r") as f:
            self.cfg = yaml.load(f, Loader = yaml.FullLoader)
   
    def add_to_log(self):
        filename = self.savepath + "/" + self.cfg['model_family'] + '-log.json'

        self.f = open(filename, 'w')
        self.f.write(json.dumps(self.log))
        self.f.flush()
        
    def run(self):
        graphs = {}
        [c, h, w] = self.cfg['input_shape']
        sample_count = self.cfg['sample_count']
        #sample_count = 2

        if self.cfg['model_family'] in MODELZOO:
            versions = [None]
            if 'modelids' in self.cfg:
                versions = self.cfg['modelids']
            sconfigs = []

            for vs in versions:
                count_index = 0
                random.seed(100)
                # tf.reset_default_graph()
                
                input_tensor = keras.Input(shape=[h, w, c], batch_size=1)
                model = MODELZOO[self.cfg['model_family']](input_tensor, self.cfg, vs, False)
                modelname = self.cfg['model_family']
                
                if vs:
                    modelname += str(vs)
                
                # print(model.sconfig)

                if not model.sconfig in sconfigs:
                    sconfigs.append(model.sconfig)
                    save_to_models(self.savepath, [input_tensor], [model.out], modelname, str(count_index))
                    self.log[modelname+"_"+str(count_index)] = model.config
                    self.add_to_log()

                count_index += 1

                while count_index < sample_count:
                    tf.reset_default_graph()
                    input_tensor = keras.Input(shape=[h, w, c], batch_size=1)
                    model = MODELZOO[self.cfg['model_family']](input_tensor, self.cfg, vs, True)
                    # print(modelname, count_index)
                    # print(model.sconfig)
                    if not model.sconfig in sconfigs:
                        sconfigs.append(model.sconfig)

                        save_to_models(self.savepath, [input_tensor], [model.out], modelname, str(count_index))
                        self.log[modelname+"_" + str(count_index)] = model.config
                        self.add_to_log()
                    
                    count_index += 1
                # for mid in self.log:
                #     for layer in self.log[mid]:
                #         print(mid, layer, self.log[mid][layer])

# nasbench201
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


