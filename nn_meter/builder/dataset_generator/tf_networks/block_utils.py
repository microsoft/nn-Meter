from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf 
from typing import Tuple, List, Union
import shutil
import os

import time 


def save_to_models(modelpath, inputs, outputs, blockname, label = ""):
    savepath = os.path.join(modelpath, "saved_model")
    create_dir(savepath)
    tfpath = os.path.join(modelpath, "tflite")
    create_dir(tfpath)
    pbpath = os.path.join(modelpath, "pb")
    create_dir(pbpath)
    if label == "":
        graphname = blockname+"_"+str(int(time.time()*1000))
    else:
        graphname = blockname+"_"+label
    savepath = os.path.join(savepath, graphname)
    create_dir(savepath)
    outputs_ops_names = [o.op.name for o in outputs]
    #print('sess run')
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

            ## save to saved_model
        #print('save to model', inputs)
        inputs_name, outputs_name = to_saved_model(
                sess, inputs, outputs, savepath, replace_original_dir = True
            )

    ## delete saved_model 
    shutil.rmtree(savepath)
    return savepath, os.path.join(tfpath, graphname+'.tflite'), os.path.join(pbpath, graphname+'.pb'), inputs_name, outputs_name

def generate_input_tensor(shapes: List[List[int]]):
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
        signature_def_map = sigs
    )

    builder.save()
    return list(inputs_dic.keys()), list(outputs_dic.keys())

def create_dir(path):
    if os.path.isdir(path) == False:
        os.mkdir(path)
