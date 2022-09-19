from dataclasses import dataclass
import os
import pathlib
from typing import List


FILE_DIR = pathlib.Path(__file__).parent.resolve()
CHANNEL_DIVISIBLE = 8
NET_IDS = ['ofa_proxyless_d234_e346_k357_w1.3', 'ofa_mbv3_d234_e346_k357_w1.0', 'ofa_resnet50']
FRAMEWORKS = ['tflite', 'openvino', 'onnx', 'tensorrt']


@dataclass
class file_paths:
	_results_dir = os.path.join(FILE_DIR, '../results')
	sample_acc_path = os.path.join(_results_dir, 'acc/D0210_onnx_eval_{}.csv')
	lut_path = os.path.join(_results_dir, 'lut/{}_{}.csv')
	latency_predict_path = os.path.join(_results_dir, 'latency/{}_{}_predict.csv')
	latency_benchmark_path = os.path.join(_results_dir, 'latency/{}_{}_benchmark.csv')
	analyse_txt_path = os.path.join(_results_dir, 'analyse/result.txt')
	latency_acc_fig_path = os.path.join(_results_dir,'analyse/latency_accuracy.png')
	torch_eval_mbv3_acc_path = os.path.join(_results_dir, 'acc/torch_eval_mbv3.csv')

def make_divisible(v, divisor=CHANNEL_DIVISIBLE, min_val=None):
	if min_val is None:
		min_val = divisor
	new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
	# Make sure that round down does not go down by more than 10%.
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v


def sample_list_from_acc_file(sample_acc_file_path: str) -> List:
	rv = []
	with open(sample_acc_file_path, 'r') as f:
		f.readline()
		for line in f.readlines():
			sample_str, *_ = line.split(',')
			rv.append(sample_str)
	return rv


def parse_sample_str(sample_str: str):
	if sample_str.startswith('d'): # resnet
		d_str, e_str, w_str = sample_str.split('_')
		d_list = [int(x) for x in d_str[1:]]
		w_list = [int(x) for x in w_str[1:]]
		e_list = [int(e_str[i:i+2]) / 100 for i in range(1, len(e_str), 2)]
		return dict(d=d_list, e=e_list, w=w_list)
	
	else: # proxylessnas or mobilenetv3
		ks_str, e_str, d_str = sample_str.split('_')
		ks_list = [int(x) for x in ks_str[2:]]
		e_list = [int(x) for x in e_str[1:]]
		d_list = [int(x) for x in d_str[1:]]
		return dict(ks=ks_list, e=e_list, d=d_list)


def net_type2id(net_type: str) -> str:
	NET_TYPE2ID = {
		'proxylessnas': 'ofa_proxyless_d234_e346_k357_w1.3',
		'mobilenetv3': 'ofa_mbv3_d234_e346_k357_w1.0', 
		'resnet': 'ofa_resnet50'
	}
	return NET_TYPE2ID[net_type]