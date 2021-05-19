import os
from utils.grapher_tool import Grapher

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_fusion_unit(name):
    filename = os.path.join(BASE_DIR, f'{name}_fusionunit.json')
    return Grapher(filename)
