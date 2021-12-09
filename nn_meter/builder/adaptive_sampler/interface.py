# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from argparse import ArgumentParser
from generator.generator_block import*

from device_utils.utils import*
from regression.build_regression_model import*
from generator.generator_block import*
from device_utils.ADBConnect import*
from device_utils.run_on_device import*
from device_utils.utils import*
from generator.generator_block import*

from silence_tensorflow import silence_tensorflow
silence_tensorflow()


# sampling from prior distribution
def init_sampler():
    
    # parsing arguments
    parser = ArgumentParser()
    parser.add_argument("--config", type=str,default="configs/conv.yaml")
   
    parser.add_argument("--rootdir", type=str, default="data")
    args=parser.parse_args()
    
  
    generator=generation(args)
    generator.setconfig_by_file(args.config)
    generator.run('prior')

   
    

# measure latency on device
def measure_latency():
    # def benchmark_model_folder(dir,filename):   ## benchmark model on various devices
    #      tfs=glob(dir+"/tflite/**.tflite")
    #      id=0
    #      adb=ADBConnect("98281FFAZ009SV")
    #      tested=[]
        
        
    #      with open(filename,'w') as f:
    #         for tf in tfs:
    #             tfmodel=tf.replace("\\","/")
    #             config=tfmodel.split("/")[-1].replace(".tflite","")
    #             std_ms,avg_ms=run_on_android(tfmodel,adb)
    #             print('here',dir+'/'+config)
    #             print(std_ms/avg_ms,avg_ms,std_ms)                        
    #             writeline=str(id)+','+str(config)+','+str(avg_ms)+','+str(std_ms)+','+str(std_ms/avg_ms*100)
    #             f.write(writeline+'\n')
    #             f.flush()                                             
    #             id+=1
            
        


    # if __name__ == "__main__":

    #     # parsing arguments
    #     parser = ArgumentParser()
    #     parser.add_argument("--datadir", type=str,default="data/conv-bn-relu-run1")
    #     parser.add_argument("--outputfile", type=str, default="test.csv")
    #     args=parser.parse_args()
    #     #args.savepath= get_output_folder('/data1/datasets/', args.savepath)
    
    #     benchmark_model_folder(args.datadir,'args.outputfile)
    pass
        



# fine-grained sampling for data with large errors
def run_adaptive_sampler():
    
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


    pass