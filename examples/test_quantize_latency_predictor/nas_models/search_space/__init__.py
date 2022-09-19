from .proxylessnas_space import ProxylessNASSpace
from .mobilenetv3_space import MobileNetV3Space
from .resnet_space import ResNetSpace

netid2space = {
    'ofa_proxyless_d234_e346_k357_w1.3': lambda: ProxylessNASSpace(width_mult=1.3),
    'ofa_mbv3_d234_e346_k357_w1.0': MobileNetV3Space,
    'ofa_resnet50': ResNetSpace
}


def search_space_block_configs(net_id, **kwargs):
    space = netid2space[net_id](**kwargs)
    return space.block_configs

def config2str(net_id, config):
    if net_id == 'ofa_proxyless_d234_e346_k357_w1.3':
        return ProxylessNASSpace.config2str(config)
    if net_id == 'ofa_mbv3_d234_e346_k357_w1.0':
        return MobileNetV3Space.config2str(config)
    if net_id == 'ofa_resnet50':
        return ResNetSpace.config2str(config)
        