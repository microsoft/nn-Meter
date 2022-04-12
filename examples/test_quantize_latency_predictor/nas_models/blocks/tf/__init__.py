import tensorflow as tf

from .proxylessnas_block import BasicBlock as ProxylessNasBasicBlock
from .mobilenetv3_block import BasicBlock as MobileNetV3BasicBlock
from .resnet_block import BasicBlock as ResNetBasicBlock

def build_block_from_config(net_id, config) -> tf.keras.Model:
    if net_id == 'ofa_proxyless_d234_e346_k357_w1.3':
        return ProxylessNasBasicBlock.build_from_config(config)
    if net_id == 'ofa_mbv3_d234_e346_k357_w1.0':
        return MobileNetV3BasicBlock.build_from_config(config)
    if net_id == 'ofa_resnet50':
        return ResNetBasicBlock.build_from_config(config)