# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os


BASE_DIR = os.path.dirname(__file__)

RULE_DIR = os.path.join(BASE_DIR, 'data/fusionrules')

BACKENDS = {
    'cortexA76cpu_tflite21': os.path.join(RULE_DIR, 'rule_cortexA76cpu_tflite21.json'),
    'adreno640gpu_tflite21': os.path.join(RULE_DIR, 'rule_adreno640gpu_tflite21.json'),
    'adreno630gpu_tflite21': os.path.join(RULE_DIR, 'rule_adreno640gpu_tflite21.json'),
    'myriadvpu_openvino2019r2': os.path.join(RULE_DIR, 'rule_myriadvpu_openvino2019r2.json'),
}
