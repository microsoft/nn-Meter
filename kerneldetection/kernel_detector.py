import sys
sys.path.append("kerneldetection")
from rulelib.rule_reader import RuleReader
from rulelib.rule_splitter import RuleSplitter
from frozenpb_parser import FrozenPbParser
from grapher_tool import Grapher
import json
import os
import pandas as pd
import argparse
import copy
from itertools import groupby
backend_maps = {
  "cpu":"tflite_cpu",
  "gpu":"tflite_gpu",
  "vpu":"vpu"
}

dummy_types = [
    'Const',
    'Identity',
    'Placeholder',
]

op_alias = {
    'Relu6': 'relu',
    'Relu': 'relu',
    'Add': 'add',
    'Biasadd': 'add',
    'Conv2D': 'conv',
    'Reshape': 'reshape',
    'FusedBatchNorm': 'bn',
    'FusedBatchNormV3': 'bn',
    'MatMul': 'fc',
    'MaxPool': 'maxpool',
    'AvgPool': 'avgpool',
    'Mean': 'gap',
    'Mul': 'mul',
    'DepthwiseConv2dNative': 'dwconv',
    'ConcatV2': 'concat',
    'Split': 'split',
    

}


fusion_map = {
    'SE': 'mul-avgpool-conv-add-relu-conv-add-add-relu-mul',
    'hswish': 'relu-mul-mul-add',
    'bn':"bnV3",
    'channelshuffle': 'reshape-Transpose-reshape-Pack-StridedSlice-Pack-StridedSlice',
     'global-avgpool': 'gap-reshape',
    
}
def get_input_tensors(node, graph):
    input_tensors = []
    for inbound in graph.get_node_inbounds(node):
        try:
            shape = graph.get_node_attr(inbound)['output_shape']
            type = graph.get_node_type(node)
            if shape and type not in dummy_types:
                if graph.get_node_type(inbound) == 'Split':
                    outbounds = graph.get_node_outbounds(inbound)
                    shapes = shape
                    for outbound, shape in zip(outbounds, shapes):
                        if outbound == node:
                            input_tensors.append(shape)
                else:
                    input_tensors.append(shape)
        except:
            pass
    return input_tensors



def bb_to_kernel(bb, graph):
    types = [graph.get_node_type(node) for node in bb]
    #print(types)
    types = [t for t in types if t and t not in dummy_types]

    for old, new in op_alias.items():
        for i in range(len(types)):
            types[i] = types[i].replace(old, new)
            

    if types:
        type = '-'.join(types)
        for block, ops in fusion_map.items():
            type = type.replace(ops, block)

        kernel = {
            'op': type,
        }

        layer = bb[0]
        type = types[0]
        attr = graph.get_node_attr(layer)['attr']
        shape = graph.get_node_attr(layer)['output_shape']
        if type in ['conv', 'dwconv']:
            weight_shape = attr['weight_shape']
            try:
                kernel['ks'] = weight_shape[0:2]
                kernel['cin'] = weight_shape[2]
                kernel['cout'] = weight_shape[3]
                kernel['strides'] = attr['strides']
                if type=='dwconv':
                    kernel['cout']=kernel['cin']
            except:
                print(bb)
        elif type in ['maxpool', 'avgpool']:
            kernel['ks'] = attr['ksize']
            kernel['cin'] = shape[3]
            kernel['cout'] = shape[3]
            kernel['strides'] = attr['strides']
        elif type == 'fc':
            kernel['cin'] = shape[1]
            kernel['cout'] = shape[1]
        elif type == 'gap':
            kernel['cin'] = shape[3]
            kernel['cout'] = shape[3]
        elif type in ['relu','hswish']:
            kernel['cin'] = shape[-1]
            kernel['cout'] = shape[-1]

        input_tensors = get_input_tensors(layer, graph)
        kernel['input_tensors'] = input_tensors   
        #print(type,input_tensors)    
        if type not in ['relu','bn', 'fc', 'reshape',  'Pack', 'StridedSlice','split']:
            input_shape = input_tensors[0]
            kernel['inputh'] = input_shape[1]
            kernel['inputw'] = input_shape[2]
        elif type in ['fc']:
            input_shape = input_tensors[0]
            kernel['cin']=input_shape[1]
    
        if type == 'split':
            kernel['split_dim'] = attr['split_dim']
            kernel['output_tensors'] = shape
        return kernel
    else:
        return None

def merge_split(graph: Grapher):
    split_nodes = [node for node in graph.get_graph().keys() if graph.get_node_type(node) == 'Split']

    for name, group in groupby(split_nodes, lambda name: name.split('/')[0]):
        group = list(group)
        group.sort(key=lambda name: name.split('/')[1])

        split_dim = graph.get_node_attr(group[0])['attr']['split_dim']
        inbounds = graph.get_node_inbounds(group[0])
        output_shapes = [graph.get_node_attr(node)['output_shape'] for node in group]

        # assert
        for i, node in enumerate(group):
            assert graph.get_node_attr(node)['attr']['idx'] == i
            assert graph.get_node_inbounds(node) == inbounds
            assert graph.get_node_attr(node)['attr']['split_dim'] == split_dim

        graph.fuse(group, 'Split', name, {
            'attr': {
                'split_dim': split_dim,
            },
            'output_shape': output_shapes,
        }, is_block=False)

def split_model_into_kernels(input_models,hardware,save_dir,rule_dir='data/fusionrules'):

    if hardware in backend_maps:
        backend=backend_maps[hardware]
    else:
        raise ValueError('Unsupported hardware')
    splitter = RuleSplitter(RuleReader())
    kernel_types = {}
    print(input_models)   
    mname=input_models.split('/')[-1].replace(".json","")
    input_models=json.load(open(input_models,'r'))

   
    

    with pd.ExcelWriter(save_dir+'/'+mname+'_result.xlsx', engine='xlsxwriter', mode='w') as writer:
              
            indexes = []
            counts = []
            kernel_types[backend] = set({})
            reader = RuleReader(rule_dir+f'/rule_{backend}.json')
            splitter = RuleSplitter(reader)
            mdicts={}
            for mid in input_models:
                model_name=mid                
                fname=mid.split('_')[0]                
                model=input_models[model_name]
                graph = Grapher(graph=model)
                merge_split(graph)
                tmp_graph = copy.deepcopy(graph)
                result = splitter.split(tmp_graph)
                bb_types = {}
                kernels = []
                for bb in result:
                    kernel = bb_to_kernel(bb, graph)
                    if kernel is not None:
                        type = kernel['op']
                        bb_types[type] = bb_types.get(type, 0) + 1
                        kernels.append(kernel)

                
                output = {model_name: kernels}

                
            
                mdicts[model_name]=kernels

                for type, count in bb_types.items():
                    kernel_types[backend].add(type)
                    indexes.append((model_name, type))
                    counts.append(count)
                #sys.exit()
                #break

            index = pd.MultiIndex.from_tuples(indexes, names=['model', 'type'])
            df = pd.DataFrame(counts, index=index, columns=['Count'])
            df.to_excel(writer, sheet_name=backend)
            kernel_types[backend] = list(kernel_types[backend])

            filename = os.path.join(save_dir, f'{hardware}_{fname}.json')
            os.makedirs(save_dir, exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as fp:
                    json.dump(mdicts, fp,indent=2)

    print(json.dumps(kernel_types))
    return kernel_types,mdicts


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware', type=str, default='cpu')
    parser.add_argument('-i', '--input_models', type=str, required=True, help='Path to input models. Either json or pb.')
    parser.add_argument('-dir', '--save_dir', type=str,  default='results', help='Default preserve the original layer names. Readable will assign new kernel names according to types of the layers.')
    parser.add_argument('-ruledir', '--rule_dir', type=str,  default='data/fusionrules', help='Default preserve the original layer names. Readable will assign new kernel names according to types of the layers.')
    #parser.add_argument('-t', '--input_type', type=str, choices=['multi-m','single-m'], default='multi-m', help='input file type: multi-m or single-m')
    #parser.add_argument('-backend', '--backend', type=str, choices=['tflite_cpu','tflite_gpu','vpu'], default='tflite_cpu', help='Default preserve the original layer names. Readable will assign new kernel names according to types of the layers.')
    args = parser.parse_args()
    split_model_into_kernels(args.input_models,args.hardware,args.save_dir,rule_dir=args.rule_dir)

