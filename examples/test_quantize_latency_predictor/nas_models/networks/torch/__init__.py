from torch import nn

from .proxylessnas import ProxylessNASNet
from .mobilenetv3 import MobileNetV3Net
from .resnet import ResNetNet


def build_network_from_sample_str(net_id, sample_str) -> nn.Module:
    if net_id == 'ofa_proxyless_d234_e346_k357_w1.3':
        return ProxylessNASNet(sample_str=sample_str, width_mult=1.3)
    if net_id == 'ofa_mbv3_d234_e346_k357_w1.0':
        return MobileNetV3Net(sample_str=sample_str)
    if net_id == 'ofa_resnet50':
        return ResNetNet(sample_str=sample_str)


def get_model_config_str_list_from_sample_str(net_id, sample_str):
    model = build_network_from_sample_str(net_id, sample_str)
    return model.model_config_str_list