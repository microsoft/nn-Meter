
import os
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import shutil
from argparse import ArgumentParser
from generator.generator_block import*
from device_utils.ADBConnect import*
from device_utils.run_on_device import*
from device_utils.utils import*
from glob import glob
import time

def benchmark_model_folder(dir,filename):   ## benchmark model on various devices
     tfs=glob(dir+"/tflite/**.tflite")
     id=0
     adb=ADBConnect("98281FFAZ009SV")
     tested=[]
     
     
     with open(filename,'w') as f:
        for tf in tfs:
            tfmodel=tf.replace("\\","/")
            config=tfmodel.split("/")[-1].replace(".tflite","")
            std_ms,avg_ms=run_on_android(tfmodel,adb)
            print('here',dir+'/'+config)
            print(std_ms/avg_ms,avg_ms,std_ms)                        
            writeline=str(id)+','+str(config)+','+str(avg_ms)+','+str(std_ms)+','+str(std_ms/avg_ms*100)
            f.write(writeline+'\n')
            f.flush()                                             
            id+=1
           
    


if __name__ == "__main__":

    # parsing arguments
    parser = ArgumentParser()
    parser.add_argument("--datadir", type=str,default="data/conv-bn-relu-run1")
    parser.add_argument("--outputfile", type=str, default="test.csv")
    args=parser.parse_args()
    #args.savepath= get_output_folder('/data1/datasets/', args.savepath)
   
    benchmark_model_folder(args.datadir,'args.outputfile)
   
    





