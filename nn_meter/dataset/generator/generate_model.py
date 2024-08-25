# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import yaml
import json
import random
from tensorflow import keras

from .networks.tf_network import *


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


class ModelGenerator:
    def __init__(self, family_config, save_path):
        random.seed(100)
        self.save_path = save_path
        self.log = {}
        with open(family_config, "r") as f:
            self.family_config = yaml.load(f, Loader = yaml.FullLoader)
        self.log_file = os.path.join(self.save_path, self.family_config['model_family'] + '-log.json')
  
    def add_to_log(self):
        self.f = open(self.log_file, 'w')
        self.f.write(json.dumps(self.log, indent=4))
        self.f.flush()
       
    def run(self):
        c, h, w = self.family_config['input_shape']
        sample_count = self.family_config['sample_count']

        if self.family_config['model_family'] not in MODELZOO:
            raise NotImplementedError

        versions = [None]
        if 'modelids' in self.family_config:
            versions = self.family_config['modelids']

        sconfigs = []
        for vs in versions:
            count_index = 0
            while count_index < sample_count:
                # generate the model with default configs from this family as the first model
                sampling = False if count_index == 0 else True

                input_tensor = keras.Input(shape=[h, w, c], batch_size=1)
                model = MODELZOO[self.family_config['model_family']](input_tensor, self.family_config, vs, sampling)

                if model.sconfig not in sconfigs:
                    # save the h5 model file
                    modelname = self.family_config['model_family']
                    if vs: modelname += str(vs)
                    output_tensor = model.out
                    keras_model = keras.Model(input_tensor, output_tensor)
                    save_name = modelname + "_" + str(count_index)
                    keras_model.save(os.path.join(self.save_path, save_name + ".h5"))
                   
                    # log the model config
                    sconfigs.append(model.sconfig)
                    self.log[modelname + "_" + str(count_index)] = model.config
                    self.add_to_log()
                    count_index += 1

# nasbench201
def build_tiny_net(config_dict):
    import subprocess
    subprocess.check_output(f'python3 build_tiny_net.py -i "{config_dict["arch_str"]}" ' +
                            f'-o {config_dict["pb_file_name"]} ' + (f'-t {config_dict["tflite_file_name"]}' if config_dict["tflite_file_name"] !=  '' else '')
                            , shell = True, stderr = open(os.devnull, 'w'))
