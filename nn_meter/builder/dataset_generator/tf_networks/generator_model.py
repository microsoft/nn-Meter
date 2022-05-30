
from .sampler import *
from .block_utils import *
import random
import tensorflow as tf
from nn_meter.builder.nn_generator.tf_networks.models import *

import yaml
import json

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
    def __init__(self, args):
        self.args = args
        self.log = {}
        with open(args.config, "r") as f:
            self.cfg = yaml.load(f, Loader = yaml.FullLoader)
   
    def add_to_log(self):
        filename = self.args.savepath + "/"+self.cfg['model_family']+'-log.json'
        #if os.path.isfile(filename) == False:
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
                tf.reset_default_graph()
                input_tensor = generate_input_tensor( [[1, h, w, c]])[0]
                model = MODELZOO[self.cfg['model_family']](input_tensor, self.cfg, vs, False)
                modelname = self.cfg['model_family']
                
                if vs:
                    modelname += str(vs)
                
                print(model.sconfig)

                if not model.sconfig in sconfigs:
                    sconfigs.append(model.sconfig)
                    _ = save_to_models(self.args.savepath, [input_tensor], [model.out], modelname, str(count_index))
                    self.log[modelname+"_"+str(count_index)] = model.config
                    self.add_to_log()

                count_index += 1

                while count_index < sample_count:
                    tf.reset_default_graph()
                    input_tensor = generate_input_tensor( [[1, h, w, c]])[0]
                    model = MODELZOO[self.cfg['model_family']](input_tensor, self.cfg, vs, True)
                    print(modelname, count_index)
                    print(model.sconfig)
                    if not model.sconfig in sconfigs:
                        sconfigs.append(model.sconfig)

                        savemodelpath, tfpath, pbpath, inputnames, outputnames = save_to_models(self.args.savepath, [input_tensor], [model.out], modelname, str(count_index))
                        self.log[modelname+"_"+str(count_index)] = model.config
                        self.add_to_log()
                    
                    count_index += 1
                for mid in self.log:
                    for layer in self.log[mid]:
                        print(mid, layer, self.log[mid][layer])
