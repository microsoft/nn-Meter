# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from .prior_distribution_sampler import *
from .finegrained_sampler import *


class BaseConfigSampler:

    def prior_config_sampling(self, sample_num):
        ''' utilize the prior data to define the configuration sampling from the prior distribution.
        '''
        pass

    def finegrained_config_sampling(self, configs, sample_num):
        ''' for data in `configs`, perform fine-grained data sampling to generate random data around the large error data.
        '''
        pass


class ConvSampler(BaseConfigSampler):

    def prior_config_sampling(self, sample_num):
        return sampling_conv(sample_num)

    def finegrained_config_sampling(self, configs, sample_num):
        return finegrained_sampling_conv(configs, sample_num)


class DwConvSampler(BaseConfigSampler):

    def prior_config_sampling(self, sample_num):
        return sampling_dwconv(sample_num)

    def finegrained_config_sampling(self, configs, sample_num):
        return finegrained_sampling_dwconv(configs, sample_num)


class PoolingSampler(BaseConfigSampler):

    def prior_config_sampling(self, sample_num):
        return sampling_pooling(sample_num)

    def finegrained_config_sampling(self, configs, sample_num):
        return finegrained_sampling_pooling(configs, sample_num)


class FCSampler(BaseConfigSampler):

    def prior_config_sampling(self, sample_num):
        # half samples have fixed cout as 1000, other samples have random cout
        return sampling_fc(int(sample_num * 0.5), fix_cout = 1000) + sampling_fc(int(sample_num * 0.5), fix_cout = False)

    def finegrained_config_sampling(self, configs, sample_num):
        return finegrained_sampling_fc(configs, sample_num)


class ConcatSampler(BaseConfigSampler):

    def prior_config_sampling(self, sample_num):
        return sampling_concats(sample_num)

    def finegrained_config_sampling(self, configs, sample_num):
        return finegrained_sampling_concats(configs, sample_num)


class CinEvenSampler(BaseConfigSampler):

    def prior_config_sampling(self, sample_num):
        return sampling_hw_cin_even(sample_num)

    def finegrained_config_sampling(self, configs, sample_num):
        return finegrained_sampling_hw_cin_even(configs, sample_num)


class GlobalAvgPoolSampler(BaseConfigSampler):

    def prior_config_sampling(self, sample_num):
        cfgs = sampling_hw_cin(sample_num)
        new_hws = [3] * (sample_num // 2 + 1) + [7] * (sample_num // 2 + 1)
        new_hws = new_hws[:len(cfgs)]
        import random; random.shuffle(new_hws)
        for cfg, hw in zip(cfgs, new_hws): cfg["HW"] = hw
        return cfgs

    def finegrained_config_sampling(self, configs, sample_num):
        return finegrained_sampling_hw_cin(configs, sample_num)


class HwCinSampler(BaseConfigSampler):

    def prior_config_sampling(self, sample_num):
        return sampling_hw_cin(sample_num)
    
    def finegrained_config_sampling(self, configs, sample_num):
        return finegrained_sampling_hw_cin(configs, sample_num)
