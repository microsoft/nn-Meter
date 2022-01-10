# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
from .fusion_lib import get_fusion_unit
from nn_meter.utils.graph_tool import ModelGraph


class RuleReader:
    rules_default = {
        "MON": 0,
        "FN": True,
    }

    multiop_blocks = ["se", "hswish", "channelshuffle", "gap"]

    def __init__(self, rule_file=None):
        self.rules = {}
        if rule_file:
            with open(rule_file, "r") as fp:
                self.rules = json.load(fp)
        self._extract_fusible()
        self._parse_multiop_block()

    def is_fusible(self, node_type, outnode_type):
        return (node_type, outnode_type) in self.fusible

    def query_rule(self, rule):
        if rule not in self.rules or self.rules[rule]["obey"] is None:
            return self.rules_default[rule]
        else:
            return self.rules[rule]["obey"]

    def _extract_fusible(self):
        def get_name(i):
            return f"{ops[i]}_{i}"

        self.fusible = []
        self.fusion_units = {}
        for name, rule in self.rules.items():
            if rule["obey"] and name.startswith("BF"):
                ops = name.split("_")[1:]
                if len(ops) == 2:
                    self.fusible.append((ops[0], ops[1]))
                elif len(ops) > 2:
                    fusion_unit = {}
                    for i in range(0, len(ops)):
                        fusion_unit[get_name(i)] = {
                            "attr": {
                                "type": ops[i],
                                "attr": {},
                            },
                            "inbounds": [get_name(i - 1)] if i > 0 else [],
                            "outbounds": [get_name(i + 1)] if i < len(ops) - 1 else [],
                        }
                    self.fusion_units["-".join(ops)] = [ModelGraph(graph=fusion_unit)]

    def _parse_multiop_block(self):
        for block in self.multiop_blocks:
            self.fusion_units[block] = get_fusion_unit(block)
