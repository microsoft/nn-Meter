import yaml
from yacs.config import CfgNode as CfgNode

_C=CfgNode()

_C.BLOCK = CfgNode(new_allowed=True)
'''
_C.BLOCK.CIN_min = 8
_C.BLOCK.CIN_max = 1024
_C.BLOCK.CSTEP_min=2

_C.BLOCK.COUT_min = 8
_C.BLOCK.COUT_max = 1024






_C.BLOCK.NODE=CfgNode(new_allowed=True)
'''
'''
_C.BLOCK.NODE=CfgNode()

_C.BLOCK.NODE.kernel_size = [3,5,7,9]
_C.BLOCK.NODE.stride=  [1,2]
_C.BLOCK.NODE.bm=[1,2,4]
_C.BLOCK.NODE.gw=[]
_C.BLOCK.NODE.se_r=[None,1/4]
#with open('configs/blocks.yaml','w') as f:
 #   strs=cfg.dump(stream=f)
 #   print(strs)
'''



def __get_cfg_defaults():
    return _C.clone()

def get_config(filename):
    cfg=__get_cfg_defaults()

    cfg.merge_from_file(filename)
    cfg.freeze()
    return cfg
#



