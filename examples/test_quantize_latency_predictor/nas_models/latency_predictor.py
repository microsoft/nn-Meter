from .networks.tf import get_model_config_str_list_from_sample_str

# Latency predictor by Xudong
class LatencyPredictor:

	def __init__(self, net_id, lut_path: str):
		self.lut = self.build_lut(lut_path)
		self.net_id = net_id

	@staticmethod
	def build_lut(lut_path):
		lut = {}
		with open(lut_path, 'r') as f:
			f.readline()
			for line in f.readlines():
				block_key, fp32_ms, int8_ms, _ = line.split(',')
				lut[block_key] = [float(fp32_ms), float(int8_ms)]
		return lut

	def query(self, key):
		return self.lut[key]

	def predict(self, sample_str: str, show_detail=False):
		fp32_ms = 0
		int8_ms = 0
		
		model_config_str_list = get_model_config_str_list_from_sample_str(self.net_id, sample_str)
		for config_str in model_config_str_list:
			this_fp32_ms, this_int8_ms = self.query(config_str)
			fp32_ms += this_fp32_ms
			int8_ms += this_int8_ms
			if show_detail:
				print(f'{config_str:<70}{fp32_ms:.2f} {this_fp32_ms:.2f}')

		return fp32_ms, int8_ms, round(fp32_ms / int8_ms, 1)
			
