import argparse
import tensorflow as tf
import numpy as np

def str2structure(xstr):
    nodestrs = xstr.split('+')
    genotypes = []
    for i, node_str in enumerate(nodestrs):
        inputs = list(filter(lambda x: x !=  '', node_str.split('|')))
        for xinput in inputs: assert len(xinput.split('~'))  ==  2, 'invalid input length : {:}'.format(xinput)
        inputs = ( xi.split('~') for xi in inputs )
        input_infos = tuple( (op, int(IDX)) for (op, IDX) in inputs)
        genotypes.append( input_infos )
    return genotypes

def flatten(_input):
    input_shape = _input.shape.as_list()
    if len(input_shape) !=  2:
        return tf.reshape(_input, [-1, np.prod(input_shape[1:])])
    else:
        return _input

def fc_layer(_input, out_channels, opname = '', use_bias = False, param_initializer = None):
    features_total = int(_input.get_shape()[-1])
    if not param_initializer:
        param_initializer = {}
    with tf.compat.v1.variable_scope(opname+'.fc'):
        init_key = '%s/weight' % tf.get_variable_scope().name
        initializer = param_initializer.get(init_key, tf.contrib.layers.xavier_initializer())
        weight = tf.compat.v1.get_variable(name = 'weight', shape = [features_total, out_channels], initializer = initializer)
        output = tf.matmul(_input, weight)
        if use_bias:
            init_key = '%s/bias' % tf.get_variable_scope().name
            initializer = param_initializer.get(init_key, tf.constant_initializer([0.0] * out_units))
            bias = tf.get_variable(name = 'bias', shape = [out_units], initializer = initializer)
            output = output + bias
    return output

def conv2d(_input, out_channels, kernel_size, stride = 1, padding = 'SAME', opname = '', param_initializer = None):
    in_features = int(_input.get_shape()[3])

    if not param_initializer:
        param_initializer = {}
    output = _input

    with tf.compat.v1.variable_scope(opname+'.conv'):
        init_key = '%s/weight' % tf.compat.v1.get_variable_scope().name
        initializer = param_initializer.get(init_key, tf.contrib.layers.variance_scaling_initializer())
        weight = tf.compat.v1.get_variable(name = 'weight', shape = [kernel_size, kernel_size, in_features, out_channels], initializer = initializer)
        output = tf.nn.conv2d(output, weight, [1, stride, stride, 1], padding = padding, data_format = 'NHWC')
    return output

def batch_norm(_input, is_training = False, opname = '', epsilon = 1e-3, decay = 0.9):

    with tf.compat.v1.variable_scope(opname+".batchnorm"):
        output = tf.contrib.layers.batch_norm(_input, scale = True, is_training = is_training, updates_collections = None, epsilon = epsilon, decay = decay, data_format = 'NHWC')
    return output

def convbnrelu(_input, out_channels, kernel_size, stride = 1, relu = False, padding = 'SAME', opname = ''):
    with tf.compat.v1.variable_scope(opname+".convbn"+('relu' if relu else '')):
        x = conv2d(_input, out_channels, kernel_size, stride)
        x = batch_norm(x)
        if(relu): x = tf.nn.relu(x)
    return x

def avgpool(_input, kernel_size, stride = 1, padding = 'VALID', opname = ''):
    with tf.compat.v1.variable_scope(opname+'.avgpool'):
        return tf.nn.avg_pool2d(_input, kernel_size, stride, padding = padding)

def reluconvbn(_input, out_channels, kernel_size, stride = 1, padding = 'SAME', opname = ''):
    with tf.compat.v1.variable_scope(opname+".reluconvbn"):
        x = tf.nn.relu(_input)
        x = conv2d(x, out_channels, kernel_size, stride)
        x = batch_norm(x)
    return x

def sepconv(_input, out_channels, kernel_size, stride = 1, padding = 'SAME', opname = ''):
    with tf.compat.v1.variable_scope(opname+".sepconv"):
        x = tf.nn.relu(_input)
        x = conv2d(x, out_channels, kernel_size, stride)
        x = conv2d(x, out_channels, 1, stride)
        x = batch_norm(x)
    return x

def dualsepconv(_input, out_channels, kernel_size, stride = 1, padding = 'SAME', opname = ''):
    x = sepconv(_input, out_channels, kernel_size, stride, opname = opname+'.1')
    x = sepconv(x, out_channels, kernel_size, 1, opname = opname+'.1')
    return x

def pooling(_input, out_channels, mode, stride = 1, padding = 'SAME', opname = ''):
    with tf.compat.v1.variable_scope(opname+".pooling"):
        tensor_shape = _input.get_shape().as_list()
        if tensor_shape[-1] !=  out_channels:
            _input = reluconvbn(_input, out_channels, 1, 1, 1, opname = opname+".pooling")
        if(mode  ==  'avg'):
            x = tf.nn.avg_pool2d(_input, 3, stride, padding = padding)
        else:
            if(mode  ==  'max'):
                x = tf.nn.max_pool2d(_input, 3, stride, padding = padding)
            else:
                raise NotImplementedError
    return x

def zero(_input, out_channels, stride, opname = ''):
    tensor_shape = _input.get_shape().as_list()
    with tf.compat.v1.variable_scope(opname+".zeros"):
        return ""
        # if tensor_shape[-1]  ==  out_channels:
        #     if stride  ==  1:
        #         return tf.math.multiply(_input, .0)
        #     else:
        #         raise NotImplementedError
        # else:
        #     tensor_shape[-1] = out_channels
        #     return tf.zeros(shape = tensor_shape, dtype = "float32") 

def skip(_input, out_channels, identity, stride, opname = ''):
    tensor_shape = _input.get_shape().as_list()
    with tf.compat.v1.variable_scope(opname+".skip"):
        if stride  ==  1 and tensor_shape[-1]  ==  out_channels:
           return _input
        else:
            if stride  ==  1:
                x = conv2d(_input, out_channels, 1, stride)
                x = batch_norm(x)
                return x
            else:
                raise NotImplementedError
    

OPS = {
  'none'         : lambda x, out_channel, stride, opname: zero(x, out_channel, stride, opname = opname), 
  'avg_pool_3x3' : lambda x, out_channel, stride, opname: pooling(x, out_channel, 'avg', stride, opname = opname), 
  'max_pool_3x3' : lambda x, out_channel, stride, opname: pooling(x, out_channel, 'max', stride, opname = opname), 
  'nor_conv_7x7' : lambda x, out_channel, stride, opname: reluconvbn(x, out_channel, 7, stride, opname = opname), 
  'nor_conv_3x3' : lambda x, out_channel, stride, opname: reluconvbn(x, out_channel, 3, stride, opname = opname), 
  'nor_conv_1x1' : lambda x, out_channel, stride, opname: reluconvbn(x, out_channel, 1, stride, opname = opname), 
  'dua_sepc_3x3' : lambda x, out_channel, stride, opname: dualsepconv(x, out_channel, 3, stride, opname = opname), 
  'dua_sepc_5x5' : lambda x, out_channel, stride, opname: dualsepconv(x, out_channel, 5, stride, opname = opname), 
  'dil_sepc_3x3' : lambda x, out_channel, stride, opname: sepconv(x, out_channel, 3, stride, opname = opname), 
  'dil_sepc_5x5' : lambda x, out_channel, stride, opname: sepconv(x, out_channel, 5, stride, opname = opname), 
  'skip_connect' : lambda x, out_channel, stride, opname: skip(x, out_channel, -1, stride, opname = opname)
}

if __name__  ==  '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-i', 
            '--input_arch_string', 
            type = str, 
            required = True)
    parser.add_argument(
            '-o', 
            '--output_file_name', 
            type = str, 
            required = True)
    parser.add_argument(
            '-t', 
            '--generate_tflite_file_name', 
            default = '', 
            type = str)
    parser.add_argument(
            '-c', 
            '--input_channel', 
            default = 16, 
            type = int)
    parser.add_argument(
            '-l', 
            '--num_of_layers', 
            default = 5, 
            type = int)
    parser.add_argument(
            '-n', 
            '--num_of_classes', 
            default = 10, 
            type = int)
    args = parser.parse_args()  
    
    block_structure = str2structure(args.input_arch_string)

    C = args.input_channel
    N = args.num_of_layers

    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N    
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

    # Input
    input_tensor = tf.compat.v1.placeholder(tf.float32, (1, 32, 32, 3))
    # Stem
    x = convbnrelu(input_tensor, args.input_channel, 3, 1, relu = False, opname = 'stem')
    # Blocks
    for index, (out_channel, stride) in enumerate(zip(layer_channels, layer_reductions)):
        if stride:
            print('>>>stride!', index, x.get_shape())
            downsample = None
            _input = x

            x = avgpool(x, 2, 2, opname = '%d.downsample' % index)
            x = conv2d(x, out_channel, 1, 1, opname = '%d.downsample' % index)
            _downsample = x

            x = convbnrelu(_input, out_channel, 3, 2, relu = True, opname = '%d.1' % index)
            x = convbnrelu(x, out_channel, 3, 1, relu = False, opname = '%d.2' % index)

            x = tf.math.add(x, _downsample)
            
        else:
            collector_nodes = []
            collector_nodes_ops = []
            collector_nodes_args = []
            node_IN = []
            node_IX = []
            for i in range(0, len(block_structure)):
                print('>>>block!', index, i, x.get_shape())
                node_info = block_structure[i]
                idx = 0
                cur_index = []
                cur_innod = []
                for (opname, op_in) in node_info:
                    idx = idx + 1
                    if(stride  ==  False):
                        stride = 1
                    print('<<<', index, i, opname, op_in, stride, x.get_shape())
                    collector_nodes.append(OPS[opname])
                    collector_nodes_ops.append(opname)
                    if op_in  ==  0:
                        collector_nodes_args.append([out_channel, stride, '%d.%d.%d' % (index, i, idx)])
                    else:
                        collector_nodes_args.append([out_channel, 1, '%d.%d.%d' % (index, i, idx)])
                    cur_index.append(len(collector_nodes) - 1)
                    cur_innod.append(op_in)

                node_IX.append( cur_index )
                node_IN.append( cur_innod )

            nodes = [x]
            out_nodes = []
            for i, (node_layers, node_innods) in enumerate(zip(node_IX, node_IN)):
                # pdb.set_trace()
                for _il, _ii in zip(node_layers, node_innods):
                    print('>>>>>', collector_nodes_ops[_il], collector_nodes_args[_il], nodes[_ii])
                    if(nodes[_ii] !=  None):
                        out_nodes.append(collector_nodes[_il](nodes[_ii], *collector_nodes_args[_il]))
                out_nodes = [ele for ele in out_nodes if ele !=  ''] 
                print('<><><>', out_nodes)

                if len(out_nodes) > 1:
                    node_feature = out_nodes[1]
                    for add_nodes in out_nodes[2:]:
                        node_feature = tf.math.add(node_feature, add_nodes)
                else:
                    if len(out_nodes)  ==  0:
                        node_feature = None
                    else:
                        node_feature = out_nodes[0]

                nodes.append(node_feature)
                out_nodes = [nodes[-1]]
            print('!!!', nodes)
            x = nodes[-1]
    

    x = batch_norm(x)
    x = tf.nn.relu(x)

    x = tf.reduce_mean(x, axis = [1, 2], keep_dims = True)
    x = flatten(x)
    x = fc_layer(x, args.num_of_classes)

    output_tensor = x

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if args.generate_tflite_file_name !=  '':
        converter = tf.lite.TFLiteConverter.from_session(sess, input_tensors = [input_tensor], output_tensors = [output_tensor])
        tflite_model = converter.convert()
        open(args.generate_tflite_file_name, 'wb').write(tflite_model)

    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), [output_tensor.op.name])
    with tf.gfile.GFile(args.output_file_name, "wb") as f:
        f.write(output_graph_def.SerializeToString())
