# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from nn_meter.builder.utils import Latency


class TestCase:
    def __init__(self, data):
        self.data = {}
        for key, value in data.items():
            pass

        pass

class TestCases:
    '''
    testcase[op] = {
                    'model': model,
                    'shapes': shapes
                }
    '''
    def __init__(self):
        self.data = {}
        self.profiled = False # check whether the test cases are profiled
        self.detected = False # check whether the test cases are detected by fusion rule
    
    def feed(self, data):
        for name, testcase in data.items():
            self.add(name, testcase)

    def add(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data[key]

    def add_latency(self, key, latency):
        if self.with_latency == False:
            self.with_latency = True
    
    def add_rule():
        pass

    def __repr__(self) -> str:
        pass
    
    def _dump(self):
        return {name: testcase._dump() for name, testcase in self.data.items()}
