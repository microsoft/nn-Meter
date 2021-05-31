# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from predictors.extract_feature import*
from predictors.kernel_predictor import*
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
import matplotlib.pyplot as plt 
import pickle,sys,os 
import argparse,time
from glob import glob

parser = argparse.ArgumentParser("predict model latency on device")
parser.add_argument('--hardware', type=str, default='cpu')
parser.add_argument('--data', type=str, default='alexnet')
parser.add_argument('--save_dir', type=str, default='outputdata/predictorzoo')


args=parser.parse_args()

if os.path.isdir(args.save_dir)==False:
  os.mkdir(args.save_dir)
  args.save_dir=args.save_dir+"/"+args.hardware
  os.mkdir(args.save_dir)


filenames=glob(args.data+"/**.csv")
for filename in filenames:
  #if 'se' in filename:
    kernelname=filename.split('/')[-1].replace(".csv","").replace("-",'')
    print(filename,kernelname)
    X,Y=read_kernel_latency(filename)
    print(len(X))
    print(X[0])
    model=get_model(args.hardware,kernelname)
    if model!=None:
      trainx, testx, trainy, testy = train_test_split(X, Y, test_size=0.2, random_state=10)
      model.fit(trainx,trainy)
      predicts=model.predict(testx)
      rmse,rmspe,error,acc5,acc10,acc15=lat_metrics(predicts,testy)
      print(rmse,rmspe,error,acc5,acc10,acc15)
      model.fit(X,Y)
      with open(args.save_dir+'/'+kernelname+'.pkl','wb') as f:
        pickle.dump(model,f)
      
        
      


    print('\n')

  
