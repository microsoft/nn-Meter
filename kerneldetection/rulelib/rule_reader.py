import json
from match_helper import MatchHelper
from fusion_lib.utils import get_fusion_unit
from grapher_tool import Grapher


class RuleReader:
    op_map = {
        'relu': 'ReLU',
        'reshape': 'Reshape',
        'conv': 'Conv2D',
        'dwconv': 'DepthwiseConv2D',
        'dense': 'FC',
        'add': 'TwoInputElementWise',
        'bn': 'BatchNorm',
    }

    rules_default = {
        'MON': 0,
        'RT': True,
        'FN': True,
    }

    multiop_blocks = ['se', 'hswish', 'channelshuffle','global-avgpool']

    def __init__(self, rule_file=None):
        self.rules = {}
        if rule_file:
            with open(rule_file, 'r') as fp:
                self.rules = json.load(fp)
        self._extract_fusible()
        self._parse_multiop_block()

    def is_fusible(self, node_type, outnode_type):
        node_base_type = MatchHelper.get_base_type(node_type)
        outnode_base_type = MatchHelper.get_base_type(outnode_type)
        return (node_base_type, outnode_base_type) in self.fusible

    def query_rule(self, rule):
        if rule not in self.rules or self.rules[rule]['obey'] is None:
            return self.rules_default[rule]
        else:
            return self.rules[rule]['obey']

    def _extract_fusible(self):
        self.fusible = []
        self.fusion_units = {}
        for name, rule in self.rules.items():
            if rule['obey'] and name.startswith('BF'):
                ops = name.split('_')[1:]
                if len(ops) == 2:
                    self.fusible.append((self.op_map.get(ops[0], ops[0]), self.op_map.get(ops[1], ops[1])))
                elif len(ops) > 2:
                    fusion_unit = {}
                    get_name = lambda i: f'{ops[i]}_{i}'
                    for i in range(0, len(ops)):
                        fusion_unit[get_name(i)] = {
                            'attr': {
                                'type': self.op_map.get(ops[i], ops[i]),
                                'attr': {},
                            },
                            'inbounds': [get_name(i - 1)] if i > 0 else [],
                            'outbounds': [get_name(i + 1)] if i < len(ops) - 1 else [],
                        }
                    self.fusion_units['_'.join(ops)] = Grapher(graph=fusion_unit)

    def _parse_multiop_block(self):
        for block in self.multiop_blocks:
            self.fusion_units[block] = get_fusion_unit(block)
