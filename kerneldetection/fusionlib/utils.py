import os
import json
from utils.grapher_tool import Grapher
from kerneldetection.utils.ir_tools import convert_nodes


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_fusion_unit(name):
    filename = os.path.join(BASE_DIR, f'{name}_fusionunit.json')
    with open(filename, 'r') as fp:
        graph = convert_nodes(json.load(fp))
    return Grapher(graph=graph)
