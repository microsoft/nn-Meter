import os


BASE_DIR = os.path.dirname(__file__)

RULE_DIR = os.path.join(BASE_DIR, 'data/fusionrules')

BACKENDS = {
    'cpu': os.path.join(RULE_DIR, 'rule_tflite_cpu.json'),
    'gpu': os.path.join(RULE_DIR, 'rule_tflite_gpu.json'),
    'vpu': os.path.join(RULE_DIR, 'rule_vpu.json'),
}
