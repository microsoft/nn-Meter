
import os
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import shutil
from argparse import ArgumentParser
from generator.generator_block import*
from glob import glob
import time







if __name__ == "__main__":

    # parsing arguments
    parser = ArgumentParser()
    parser.add_argument("--config", type=str,default="configs/conv.yaml")
   
    parser.add_argument("--rootdir", type=str, default="data")
    args=parser.parse_args()
    
  
    generator=generation(args)
    generator.setconfig_by_file(args.config)
    generator.run('prior')

   
    





