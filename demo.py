


from Latencypredictor.predictors.predict_by_kernel import*
from kerneldetection.kernel_detector import*
import pickle,sys,os 
import argparse 
parser = argparse.ArgumentParser("predict model latency on device")
parser.add_argument('--hardware', type=str, default='cpu')
parser.add_argument('--mf', type=str, default='alexnet')
parser.add_argument('--level', type=str, default='kernel')
parser.add_argument('--input_models', type=str, required=True, help='Path to input models. Either json or pb.')
parser.add_argument( '--save_dir', type=str,  default='results', help='Default preserve the original layer names. Readable will assign new kernel names according to types of the layers.')
parser.add_argument( '--rule_dir', type=str,  default='data/fusionrules', help='Default preserve the original layer names. Readable will assign new kernel names according to types of the layers.')


args=parser.parse_args()
hardware=args.hardware
input_models=args.input_models
for hardware in ['cpu','gpu','gpu1','vpu']:
    print('current hardware',hardware)
    if hardware=='gpu1':
        hw='gpu'
    else:
        hw=hardware
    kernel_types,kernel_with_features=split_model_into_kernels(input_models,hw,args.save_dir)
   # hardware=args.hardware
    #sys.exit()
    print(kernel_with_features)
    if args.level=='kernel':    
        #kernel_file='data/model_kernels/'+hardware+'_'+args.mf+"s.json"
        latency_file="data/model_latency/"+hardware+"/"+args.mf+"-log.csv"
        print(latency_file)
        rmse,rmspe,error,acc5,acc10=main_kernel_predict(hardware,args.mf,kernel_with_features,latency_file)






