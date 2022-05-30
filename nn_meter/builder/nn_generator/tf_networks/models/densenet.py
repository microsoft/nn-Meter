import tensorflow as tf 
from .ops import  *
from .utils import * 


class DenseNet(object):
    def __init__(self,x,cfg,version=18,sample=False,enable_out=False):  ## change channel number, kernel size
        self.input=x
        self.num_classes=cfg['n_classes']
        self.enable_out=enable_out

        #print(self.bes)
        #print(self.bcs)
        self.r=[6,12,24,16]
        self.bcs=[32] * sum(self.r)
        self.bks=[3] * sum(self.r)

        #print(cfg['sample_space'])
        
        self.cs=get_sampling_channels(cfg['sample_space']['channel']['start'],cfg['sample_space']['channel']['end'],cfg['sample_space']['channel']['step'],len(self.bcs))
        self.ks=get_sampling_ks(cfg['sample_space']['kernelsize'],len(self.bks))

        #print(self.cs)
        #print(self.ks)
        self.config={}
        if sample==True:
            self.ncs=[int(self.bcs[index] * self.cs[index]) for index in range(len(self.bcs))]
            self.nks=self.ks
           
        else:
            self.ncs=self.bcs 
            self.nks=self.bks 
           
        self.sconfig='_'.join([str(x) for x in self.nks]) + '-' + '_'.join([str(x) for x in self.ncs])
        self.out=self.build()

    def add_to_log(self,op,cin,cout,ks,stride,layername,inputh,inputw):
        self.config[layername]={}
        self.config[layername]['op']=op
        self.config[layername]['cin']=cin
        self.config[layername]['cout']=cout
        self.config[layername]['ks']=ks
        self.config[layername]['stride']=stride 
        self.config[layername]['inputh']=inputh
        self.config[layername]['inputw']=inputw 

    def addblock_to_log(self,op,cin,cout,ks,stride,layername,inputh,inputw,es):
        self.config[layername]={}
        self.config[layername]['op']=op
        self.config[layername]['cin']=cin
        self.config[layername]['cout']=cout
        self.config[layername]['ks']=ks
        self.config[layername]['stride']=stride 
        self.config[layername]['inputh']=inputh
        self.config[layername]['inputw']=inputw 
        self.config[layername]['es_channel']=es

    def build(self):
        x=conv2d(self.input,64,7,opname='conv1',stride=2,padding='SAME') #def conv2d(_input,out_features,kernel_size,opname='',stride=1,padding='SAME',param_initializer=None):
        x=batch_norm(x,opname='conv1.bn')
        x=activation(x,'relu',opname='conv1.relu')
       # print(x.shape)

        self.add_to_log('conv-bn-relu',3,64,7,2,'layer1',self.input.shape.as_list()[1],self.input.shape.as_list()[2])

        (h,w)=x.shape.as_list()[1:3]

        x=max_pooling(x,3,2,opname='conv1')
        self.add_to_log('max-pool',64,64,3,2,'layer2',h,w)  ## bug: input size error
        #print(x.shape)
        layercount=3
        lastchannel=64

        index=0

        for layers in self.r:
            for l in range(layers):
                x,log=dense_layer(x,self.ncs[index],self.nks[index],name='layer' + str(layercount),log=True)
                self.config.update(log)
                #print(layercount,x.shape)
                index +=1 
                layercount +=1
            ## transition layer 
            (h,w,cin)=x.shape.as_list()[1:4]

            x=batch_norm(x,opname='conv' + str(layercount) + '.1')
            x=activation(x,'relu',opname='conv' + str(layercount) + '.1')
            self.add_to_log('bn-relu',cin,cin,None,None,'layer' + str(layercount),h,w)

            layercount +=1
            x=conv2d(x,cin//2,1,opname='conv' + str(layercount) + '.1',stride=1)
            self.add_to_log('conv',cin,cin//2,1,1,'layer' + str(layercount),h,w)
            layercount +=1
            (h1,w1,cin1)=x.shape.as_list()[1:4]
            x=avg_pooling(x,2,2,opname='conv' + str(layercount))           
            self.add_to_log('avg-pool',cin1,cin1,2,2,'layer' + str(layercount),h1,w1)  
            layercount +=1
            #print(x.shape)

        (h,w,cin)=x.shape.as_list()[1:4]
        x = tf.reduce_mean(x, axis=[1, 2],keep_dims=True)
        x=flatten(x)
        self.add_to_log('global-pool',cin,cin,None,None,'layer' + str(layercount),1,1)
        #print(x.shape)

        x=fc_layer(x,self.num_classes,opname='fc' + str(layercount + 1))
        self.add_to_log('fc',cin,self.num_classes,None,None,'layer' + str(layercount + 1),None,None)
        #print(x.shape)

        return x