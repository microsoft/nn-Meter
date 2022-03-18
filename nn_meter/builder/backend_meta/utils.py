# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import math
import copy
from typing import List


class ProfiledResults:
    def __init__(self, results=None):
        """
        Initialize the profiled results of test models running on backends.
        
        @params:
        
        results: Dict
            The profiled results, with Dict key to be the metric name. Metrics 
            include: latency, peak/average power, energy, memory, etc.
        """
        self.data = {}
        for metric, value in results.items():
            self.data[metric] = value
    
    def set(self, metric, value):
        """ Set metric value by its name
        """
        self.data[metric] = value

    def get(self, metrics):
        ''' Get metric value by calling the name of the metric.
        '''
        if not isinstance(metrics, List):
            metrics = [metrics]
        result = {}
        for metric in metrics:
            if metric in self.data:
                result[metric] = self.data[metric]
            else:
                raise AttributeError(f"Unsupported metric {metric}.")
        return result
    
    def _dump(self):
        return {metric: str(value) for metric, value in self.data}


class Latency:
    def __init__(self, avg=0, std=0):
        if isinstance(avg, str):
            avg, std = avg.split('+-')
            self.avg = float(avg)
            self.std = float(std)
        elif isinstance(avg, Latency):
            self.avg, self.std = avg.avg, avg.std
        else:
            self.avg = avg
            self.std = std

    def __str__(self):
        return f'{self.avg} +- {self.std}'

    def __add__(self, rhs):
        if isinstance(rhs, Latency):
            return Latency(self.avg + rhs.avg, math.sqrt(self.std ** 2 + rhs.std ** 2))
        else:
            return Latency(self.avg + rhs, self.std)
    
    def __radd__(self, lhs):
        return self.__add__(lhs)

    def __mul__(self, rhs):
        return Latency(self.avg * rhs, self.std * rhs)

    def __rmul__(self, lhs):
        return self.__mul__(lhs)

    def __le__(self, rhs):
        return self.avg < rhs.avg
    
    def __gt__(self, rhs):
        return self.avg > rhs.avg

    def __neg__(self):
        return Latency(-self.avg, -self.std)

    def __sub__(self, rhs):
        return self + rhs.__neg__()


def dump_profiled_results(results, detail = False, metrics = ["latency"]):
    ''' convert Latency instance to string and return profiled results

    @params

    detail: if False, only metrics result will be dumped to the profiled results. Otherwise models information
        will be dumpled, too.
    '''
    dumped_results = {}
    for module_key, module in results.items():
        dumped_results[module_key] = {}
        for model_key, model in module.items():
            dumped_results[module_key][model_key] = {}
            if detail:
                for info_key, info in model.items():
                    if info_key == 'latency':
                        dumped_results[module_key][model_key]['latency'] = str(model['latency'])
                    else:
                        dumped_results[module_key][model_key][info_key] = info
            else:
                for info_key, info in model.items():
                    if info_key in metrics:
                        if info_key == 'latency':
                            dumped_results[module_key][model_key]['latency'] = str(model['latency'])
                        else:
                            dumped_results[module_key][model_key][info_key] = model[info_key]
    return dumped_results


def read_profiled_results(results):
    results_copy = copy.deepcopy(results)
    for item in results_copy.values():
        for model in item.values():
            if 'latency' in model:
                model['latency'] = Latency(model['latency'])
    return results_copy
