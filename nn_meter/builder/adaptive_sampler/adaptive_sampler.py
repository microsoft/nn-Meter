
import os
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import shutil
from argparse import ArgumentParser
from generator.generator_block import*
from glob import glob
from device_utils.utils import*
import time
from regression.build_regression_model import*




if __name__ == "__main__":

    # parsing arguments
    parser = ArgumentParser()
    parser.add_argument("--kernel", type=str,default="conv-bn-relu")
    parser.add_argument("--rootdir", type=str, default="data")  ## where to save model files
    parser.add_argument("--sample_count", type=int, default=10)## for each large-error-data, we sample😊
    parser.add_argument("--iteration", type=int, default=10)
    args=parser.parse_args()
   
    acc10,cfgs=build_predictor('cpu','kernel_latency',args.kernel,large_error_threshold=0.2)## use current sampled data to build regression model, and locate data with large errors in testset
    print('cfgs',cfgs)
    ### for data with large-errors, we conduct fine-grained data sampling in the channel number dimensions
    generator=generation(args)
    generator.setconfig(args.kernel,args.sample_count,cfgs)
    generator.run('finegrained')




   
    





