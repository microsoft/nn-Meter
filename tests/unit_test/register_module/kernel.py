import tensorflow.keras as keras
from nn_meter.builder.nn_generator import BaseBlock
from nn_meter.builder.kernel_predictor_builder import BaseFeatureParser, BaseConfigSampler


class MyKernel(BaseBlock):
    ''' This kernel is built by Conv, BN, and Relu layer, which is the same as the builtin `conv-bn-relu` block.
    '''
    def __init__(self, config):
        self.config = config
        self.input_shape = [config["HW"], config["HW"], config["CIN"]]
        self.input_tensor_shape = [self.input_shape]

    def get_model(self):
        class Model(keras.Model):
            def __init__(self, cout, kernel_size, strides):
                super().__init__()
                self.conv = keras.layers.Conv2D(
                    cout,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="same"
                )
                self.bn = keras.layers.BatchNormalization()
                self.relu = keras.layers.ReLU()

            def call(self, inputs):
                x = self.conv(inputs)
                x = self.bn(x)
                x = self.relu(x)
                return x

        return Model(self.config["COUT"], self.config["KERNEL_SIZE"], self.config["STRIDES"])


class MySampler(BaseConfigSampler):
    ''' This sampler is for Conv related sampler. In `prior_config_sampling` method, all configs are sampled based on existing conv model. In
    `finegrained_config_sampling` method, only CIN and COUT are sampled around the configs in parameter `configs`.
    '''

    def prior_config_sampling(self, sample_num):
        new_hws = ...
        new_cins = ...
        new_couts = ...
        new_kernel_sizes = ...
        new_strides = ...
        ncfgs = []
        for hw, cin, cout, kernel_size, stride in zip(new_hws, new_cins, new_couts, new_kernel_sizes, new_strides):
            c = {
                'HW': hw,
                'CIN': cin,
                'COUT': cout,
                'KERNEL_SIZE': kernel_size,
                'STRIDES': stride,
            }
            ncfgs.append(c)
        return ncfgs
    
    def finegrained_config_sampling(self, sample_num, configs):
        ncfgs = []
        for cfg in configs:
            cins = ...
            couts = ...
            for cin, cout in zip(cins, couts):
                c = {
                    'HW': cfg['HW'],
                    'CIN': cin,
                    'COUT': cout,
                    'KERNEL_SIZE': cfg['KERNEL_SIZE'],
                    'STRIDES': cfg['STRIDES'],
                }
                ncfgs.append(c)
        return ncfgs


class MyParser(BaseFeatureParser):
    ''' This parser utilized config "HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES", as well as the flops and parameter number as feature.
    '''
    def __init__(self, kernel_type):
        self.kernel_type = kernel_type
        self.needed_config = ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"]

    def get_feature_by_config(self, config_dict):
        feature = [config_dict[data] for data in self.needed_config]
        hw, cin, cout, kernel_size, stride = config_dict["HW"], config_dict["CIN"], config_dict["COUT"], \
            config_dict["KERNEL_SIZE"], config_dict["STRIDES"]
        param = cout * (kernel_size * kernel_size * cin + 1)
        flop = 2 * hw / stride * hw / stride * param

        flop /= 2e6
        param /= 1e6
        feature.extend([flop, param])
        return feature

    def get_config_by_feature(self, feature):
        # remove flops and params num feature from feature vector
        feature = feature[:-2]
        assert len(self.needed_config) == len(feature)
        config = {k: v for k, v in zip(self.needed_config, feature)}
        return config
