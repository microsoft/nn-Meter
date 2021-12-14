# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
from data_sampler.block_sampler import *


def get_output_folder(parent_dir, run_name):
    """
    Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    run_name: str
      Name of the run

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok = True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, run_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok = True)
    return parent_dir


import os
import time 
import shutil
from typing import List, Union
import tensorflow as tf 

from silence_tensorflow import silence_tensorflow
silence_tensorflow()


def patch_frozen_graph(graph):
    for node in graph.node:
        if 'explicit_paddings' in node.attr.keys():
            # print('Find explicit_paddings in node %s, removing.' % node.name)
            del node.attr['explicit_paddings']
        if node.op == 'AddV2':
            # print('Find AddV2 in node %s, patching to Add.' % node.name)
            node.op = 'Add'
        if node.op == 'FusedBatchNormV3':
            # print('Find FusedBatchNormV3 in node %s, patching to FusedBatchNorm.' % node.name)
            node.op = 'FusedBatchNorm'
            del node.attr['U']
    return graph


def save_to_models(modelpath, inputs, outputs, blockname, label = ""):
    savepath = os.path.join(modelpath,"saved_model")
    os.makedirs(savepath, exist_ok = True)
    tfpath = os.path.join(modelpath, "tflite")
    os.makedirs(tfpath, exist_ok = True)
    pbpath=os.path.join(modelpath, "pb")
    os.makedirs(pbpath, exist_ok = True)
    if label == "":
        graphname = blockname + "_" + str(int(time.time() * 1000))
    else:
        graphname = blockname + "_" + label
    savepath = os.path.join(savepath,graphname)
    os.makedirs(savepath, exist_ok = True)
    outputs_ops_names = [o.op.name for o in outputs]
    # with tf.compat.v1.Session() as sess:
    #     sess.run(tf.compat.v1.global_variables_initializer())

    #     # save to saved_model
    #     inputs_name, outputs_name = to_saved_model(
    #         sess, inputs, outputs, savepath,replace_original_dir = True)
    #     # save to pb
    #     constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
    #         sess, sess.graph_def, outputs_ops_names)
    #     constant_graph = patch_frozen_graph(constant_graph)
    #     with tf.gfile.FastGFile(os.path.join(pbpath,graphname + '.pb'), mode = 'wb') as f:
    #         f.write(constant_graph.SerializeToString())

    # # save to tflite
    # tfconvert_command = "tflite_convert --saved_model_dir=" + savepath +  " --output_file=" + os.path.join(tfpath,graphname + '.tflite')
    # print(tfconvert_command)
    # subprocess.call(tfconvert_command, stdout = open(os.devnull, 'w'), stderr = subprocess.STDOUT, shell = True) ## convert to tflite in a silent way

    # # delete saved_model 
    # shutil.rmtree(savepath)
    # return savepath, os.path.join(tfpath,graphname + '.tflite'), os.path.join(pbpath,graphname + '.pb'), inputs_name, outputs_name


def generate_input_tensor(shapes):
        """Generate input tensor according to given shape
        """
        tensors = [
            tf.placeholder(
                name = "input_im_{}".format(i),
                dtype = tf.float32,
                shape = shape
            ) for i, shape in zip(range(len(shapes)), shapes)
        ]
        return tensors


def to_saved_model(
    sess,
    inputs: List[Union[tf.Tensor, tf.Operation]],
    outputs: List[Union[tf.Tensor, tf.Operation]],
    path: str,
    replace_original_dir: bool
):
    from tensorflow.python.saved_model import signature_constants
    from tensorflow.python.saved_model import tag_constants

    if replace_original_dir:
        if os.path.isdir(path):
            shutil.rmtree(path)

    ## set input/output nodename
    inputs_dic = {
        "input_{}".format(idx): i if isinstance(i, tf.Tensor) else i.outputs[0]
        for idx, i in zip(range(len(inputs)), inputs)
    }
   
    outputs_dic = {
        "output_{}".format(idx): o if isinstance(o, tf.Tensor) else o.outputs[0]
        for idx, o in zip(range(len(outputs)), outputs)
    }
    #print('here1')
    

    
    saved_model = tf.compat.v1.saved_model

    builder = saved_model.builder.SavedModelBuilder(path)
    sigs = {}
    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        saved_model.signature_def_utils.predict_signature_def(
        inputs_dic, outputs_dic
    )
    builder.add_meta_graph_and_variables(
        sess,
        [tag_constants.SERVING],
        signature_def_map=sigs
    )

    builder.save()
    return list(inputs_dic.keys()),list(outputs_dic.keys())



def get_op_output_shape(op_name, input_shape):
    if op_name in ["conv", "dwconv", "convtrans", "se", "dense", "relu", "hswish", "add"]:
        return input_shape
    elif op_name == "pooling":
        return [int(input_shape[0] / 2), int(input_shape[1] / 2), input_shape[2]]
    elif op_name == "reshape":
        if len(input_shape) == 3:
            output_shape = [input_shape[2], input_shape[0], input_shape[1]]
        else:
            output_shape = [1, 2, int(input_shape[0] / 2)]
        return output_shape
    elif op_name == "concat":
        if len(input_shape) == 3:
            output_shape = [input_shape[0], input_shape[1], input_shape[2] * 2]
        else:
            output_shape = [input_shape[0] * 2]
        return output_shape
    else:
        raise ValueError(f"Unsupported operator name: {op_name} in rule-tester.")
    

def get_op_is_two_inputs(op_name):
    if op_name in ["conv", "dwconv", "convtrans", "pooling", "se", "dense", "relu", "hswish", "reshape"]:
        return False
    elif op_name in ["add", "concat"]:
        return True
    else:
        raise ValueError(f"Unsupported operator name: {op_name} in rule-tester.")
