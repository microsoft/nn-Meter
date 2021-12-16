from build_regression_model import build_predictor
from ..data_sampler.generator import Generator
class AdaptiveSampler:
    def __init__(self):
        pass
    
    # sampling from prior distribution
    def init_sampler(block_name, ):
        
        # parsing arguments   
        generator = Generator(block_name)
        generator.run('prior')

    def run_adaptive_sampler(block_name, worksapce, sample_num, iteration):

        # use current sampled data to build regression model, and locate data with large errors in testset
        acc10, cfgs = build_predictor('cpu','kernel_latency',block_name,large_error_threshold=0.2)
        print('cfgs', cfgs)
        ### for data with large-errors, we conduct fine-grained data sampling in the channel number dimensions
        generator = Generator()
        generator.setconfig(block_name, sample_num, cfgs)
        generator.run('finegrained')
