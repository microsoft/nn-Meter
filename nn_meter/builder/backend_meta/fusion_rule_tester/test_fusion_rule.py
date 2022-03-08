# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import networkx as nx
from .generate_testcase import generate_testcases
from nn_meter.builder import builder_config

config = builder_config.get_module('ruletest')


class FusionRuleTester:
    def __init__(self):
        self._testcases = generate_testcases()

    def _build_dep_dag(self):
        dag = nx.DiGraph()

        for name, cls in self._testcases.items():
            dag.add_node(name)
            for dep in cls.deps:
                dag.add_edge(dep, name)

        self._dag = list(nx.topological_sort(dag))

    def generate(self):
        testcases = {}

        for name, cls in self._testcases.items():
            testcases[name] = cls(config).save_testcase()

        return testcases

    def analyze(self, profile_results):
        self._build_dep_dag()
        result = {}

        for name in self._dag:
            if name not in profile_results:
                continue

            result[name] = {}
            rule_cls = self._testcases[name]

            obey = True
            for dep, expect in rule_cls.deps.items():
                if result[dep]['obey'] != expect:
                    obey = False

            if obey:
                rule = rule_cls(config)
                rule.load_latency(profile_results[name])
                obey = rule.test()
                if config['DETAIL']:
                    latency = {key: str(value) for key, value in rule.latency.items()}
                    result[name]['latency'] = latency

            result[name]['obey'] = obey

        return result
