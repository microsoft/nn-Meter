# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import re
import sys

import tensorflow as tf
from google.protobuf import text_format
from tensorflow import io


def dec2octpkg4(input):
    octPkgStr = ''
    for i in range(4):
        octPkgStr = octPkgStr + oct((input >> (i * 8)) %
                                    256).replace('0o', '\\')
    return octPkgStr


KEEP_DIM_PATCH =\
    '''
  attr {
    key: "keep_dims"
    value {
      b: {KEEP_DIM}
    }
  }
'''

REDUCE_DIM_PATCH =\
    '''
node {
  name: "reshape/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "{SHAPE}"
      }
    }
  }
}

node {
  name: "reshape/Reshape"
  op: "Reshape"
  input: "{INPUT_TENSOR_NAME}"
  input: "reshape/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
'''

SWISH_PATCH =\
    '''
node {
  name: "{NAME}/Sigmoid"
  op: "Sigmoid"
  input: "{INPUT}"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}

node {
  name: "{NAME}/mul"
  op: "Mul"
  input: "{INPUT}"
  input: "{NAME}/Sigmoid"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
'''

MEAN_PATCH =\
    '''
node {
  name: "{NAME}"
  op: "AvgPool"
  input: "{INPUT}"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
  attr {
    key: "ksize"
    value {
      list {
        i: 1
        i: {KERNEL_SIZE}
        i: {KERNEL_SIZE}
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "VALID"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
'''

EXPLICT_PAD_ATTR_REGEX = r'attr {\n[\s]+key: "explicit_paddings"\n[\s]'\
    r'+value {\n[\s]+list {\n[\s]+}\n[\s]+}\n[\s]+}'

U_KEY_ATTR_REGEX = r'attr {\n[\s]+key: "U"\n[\s]+value {\n[\s]+type: DT_FLOAT\n[\s]+}\n[\s]+}'

OUTPUT_SHAPE_REGEX_3 = r'([\s]+attr {\n[\s]+key: "_output_shapes"\n[\s]+value {\n[\s]+list {\n['\
    r'\s]+)(shape[\s]+{[\s]+([\s]+(dim[\s]+{\s+size:[\s]+[0-9]+[\s]+})|([\s]+'\
    r'unknown_rank: \w+([\s]+})+))+([\s]+}[\s]+)+)+([\s]})+'

OUTPUT_SHAPE_REGEX_1 = r'attr {\n[\s]+key: "_output_shapes"\n[\s]+value {\n[\s]+list {\n[\s]+shape '\
    r'{\n[\s]+dim {\n[\s]+size: [0-9]+\n[\s]+}\n[\s]+dim {\n[\s]+size: [0-9]+\n[\s]+}'\
    r'\n[\s]+dim {\n[\s]+size: [0-9]+\n[\s]+}\n[\s]+dim {\n[\s]+size: [0-9]+\n[\s]+}\n'\
    r'[\s]+}\n[\s]+shape {\n[\s]+dim {\n[\s]+size: [0-9]+\n[\s]+}\n[\s]+}\n[\s]+shape'\
    r' {\n[\s]+dim {\n[\s]+size: [0-9]+\n[\s]+}\n[\s]+}\n[\s]+shape {\n[\s]+dim {\n[\s'\
    r']+size: [0-9]+\n[\s]+}\n[\s]+}\n[\s]+shape {\n[\s]+dim {\n[\s]+size: [0-9]+\n[\s]'\
    r'+}\n[\s]+}\n[\s]+shape {\n[\s]+unknown_rank: true\n[\s]+}\n[\s]+}\n[\s]+}\n[\s]+}'

OUTPUT_SHAPE_REGEX_2 = r'[\s]+attr {\n[\s]+key: "_output_shapes"\n[\s]+value {\n[\s]+list {\n[\s]+shape '\
    r'{([\s]+dim[\s]+{\n[\s]+size:[\s]+[0-9]+\n[\s]+}\n)+([\s]+}\n)+'

REDUCTION_IDENCE_REGEX = r'node[\s]+{\n[\s]+name:[\s]+\"[^"]+reduction_indices'\
    r'\"\n[\s]+op:[\s]+\"Const\"[\s]+[\s]+attr[\s]+{\n[\s]+k'\
    r'ey:[\s]+\"dtype\"\n[\s]+value[\s]+{\n[\s]+type:[\s]+DT'\
    r'_INT32\n[\s]+}\n[\s]+}\n[\s]+attr[\s]+{\n[\s]+key:[\s]'\
    r'+\"value\"\n[\s]+value[\s]+{\n[\s]+tensor[\s]+{\n[\s]+'\
    r'dtype:[\s]+DT_INT32\n[\s]+tensor_shape[\s]+{\n[\s]+dim'\
    r'[\s]+{\n[\s]+size:[\s]+2\n[\s]+}\n[\s]+}\n[\s]+tensor_'\
    r'content:[\s]+\"\\001\\000\\000\\000\\002\\000\\000\\00'\
    r'0\"\n+([\s]+}\n[\s]+)(}\n[\s]+}\n})'


def pbtxt_processing(content):
    if content.find('explicit_paddings') != -1:
        print('Find unsupported attr: explicit_paddings, removing...\n')
        content = re.sub(EXPLICT_PAD_ATTR_REGEX, '', content)

    meanCounter = content.count('op: "Mean"')
    if meanCounter > 1:  # May not work
        endMeanNodeName = ""
        print(f'Find semi-supported op: Mean presented {meanCounter} times in pb, patching...')
        while True:
            meanOpLoc = content.find('op: "Mean"')
            if meanOpLoc == -1:
                break
            nodeNameLoc = content.rfind('name', 0, meanOpLoc)
            nodeNameDLoc = content.find('"', nodeNameLoc)
            nodeName = content[nodeNameDLoc +
                               1:content.find('"', nodeNameDLoc + 1)]

            nodeInputLoc = content.find('input', meanOpLoc)
            nodeInputDLoc = content.find('"', nodeInputLoc)
            nodeInputName = content[nodeInputDLoc +
                                    1:content.find('"', nodeInputDLoc + 1)]

            inputNodeNameLoc = content.find(f'name: "{nodeInputName}"')
            inputNodeEnd = content.find('node', inputNodeNameLoc)
            inputNodeShape = re.findall(
                r'[\d]+\n', content[inputNodeNameLoc:inputNodeEnd])

            if len(inputNodeShape) != 4:
                print(
                    f'Unexpected happened in shape inference, infered shape: {inputNodeShape} in node [{nodeName}]')
                sys.exit(-1)

            for i in range(len(inputNodeShape)):
                inputNodeShape[i] = int(inputNodeShape[i].replace('\n', ''))
            print(
                f'Found Node name: {nodeName}, Input Shape: {inputNodeShape}\nPatching the Mean operator...')

            nodeStart = content.rfind('{', 0, nodeNameLoc)
            nodeEnd = content.find('node', nodeNameLoc)

            if content[nodeStart:nodeEnd].find(
                    KEEP_DIM_PATCH.replace('{KEEP_DIM}', 'false')) != -1:
                print('Find reduce mean at top, ignore and break.')
                endMeanNodeName = nodeName
                break

            print(f'Generating the patcher, node input: {nodeInputName}')
            patcher = MEAN_PATCH.replace('{NAME}', nodeName)
            patcher = patcher.replace('{INPUT}', nodeInputName)
            patcher = patcher.replace('{KERNEL_SIZE}', str(inputNodeShape[1]))

            print('Inserting patch and removing the Mean node...\n')
            content = content[:content.rfind(
                'node', 0, nodeStart)] + patcher + content[nodeEnd:]

        print('Removing unused const.\n')
        content = re.sub(REDUCTION_IDENCE_REGEX, '', content)

        while True:
            indecOpLoc = content.find('reduction_indices')
            if indecOpLoc == -1:
                break
            indecNameLoc = content.rfind('name', 0, indecOpLoc)
            indecStart = content.rfind('{', 0, indecNameLoc)
            indecEnd = content.find('node', indecNameLoc)
            if content[indecStart:indecEnd].find(endMeanNodeName) != -1:
                break
            content = content[:content.rfind(
                'node', 0, indecStart)] + content[indecEnd:]

    if content.find('AddV2') != -1:
        print('Find unsupported op: AddV2, patching...\n')
        content = content.replace('AddV2', 'Add')

    if content.find(KEEP_DIM_PATCH.replace('{KEEP_DIM}', 'false')) != -1:
        print('Find unsupported op: reduce_dim=false, patching...')

        while True:
            keepDimLoc = content.find(
                KEEP_DIM_PATCH.replace(
                    '{KEEP_DIM}', 'false'))
            if keepDimLoc == -1:
                break

            nodeNameLoc = content.rfind('name', 0, keepDimLoc)
            nodeNameDLoc = content.find('"', nodeNameLoc)
            nodeName = content[nodeNameDLoc +
                               1:content.find('"', nodeNameDLoc + 1)]
            print(
                f'Found Node name: {nodeName}, Output Shape: {OUTPUT_FILTER}, Oct: {dec2octpkg4(OUTPUT_FILTER)}')
            print('Patching the Mean operator...')

            nodeEnd = content.find('node', nodeNameLoc)
            content = content.replace(
                f'input: "{nodeName}"',
                'input: "reshape/Reshape"')

            patcher = REDUCE_DIM_PATCH.replace('{INPUT_TENSOR_NAME}', nodeName)
            patcher = patcher.replace(
                '{SHAPE}', f'\\377\\377\\377\\377{dec2octpkg4(OUTPUT_FILTER)}')

            content = content[:nodeEnd] + patcher + content[nodeEnd:]

            content = content.replace(
                KEEP_DIM_PATCH.replace(
                    '{KEEP_DIM}', 'false'), KEEP_DIM_PATCH.replace(
                    '{KEEP_DIM}', 'true'))
            print('Modified reduce_dim=true...\n')

    if content.find('FusedBatchNormV3') != -1:
        print('Find unsupported op: FusedBatchNormV3, patching...\n')
        content = content.replace('FusedBatchNormV3', 'FusedBatchNorm')
        content = re.sub(U_KEY_ATTR_REGEX, '', content)
        content = re.sub(OUTPUT_SHAPE_REGEX_1, '', content)
        content = re.sub(OUTPUT_SHAPE_REGEX_2, '', content)

    if content.find('op: "swish_f32"') != -1:
        print('Find unsupported op: swish_f32, patching...')
        while True:
            swishOpLoc = content.find('op: "swish_f32"')
            if swishOpLoc == -1:
                break
            nodeNameLoc = content.rfind('name', 0, swishOpLoc)
            nodeNameDLoc = content.find('"', nodeNameLoc)
            nodeName = content[nodeNameDLoc +
                               1:content.find('"', nodeNameDLoc + 1)]

            nodeInputLoc = content.find('input', swishOpLoc)
            nodeInputDLoc = content.find('"', nodeInputLoc)
            nodeInputName = content[nodeInputDLoc +
                                    1:content.find('"', nodeInputDLoc + 1)]

            print(
                f'Found Node name: {nodeName}\nPatching the swish_f32 operator...')

            nodeStart = content.rfind('{', 0, nodeNameLoc)
            nodeEnd = content.find('node', nodeNameLoc)

            print(f'Generating the patcher, node input: {nodeInputName}')
            patcher = SWISH_PATCH.replace('{NAME}', nodeName)
            patcher = patcher.replace('{INPUT}', nodeInputName)

            print('Inserting patch and removing the swish_f32 node...')
            content = content[:content.rfind(
                'node', 0, nodeStart)] + patcher + content[nodeEnd:]

            print('Reconnecting the graph...\n')
            content = content.replace(
                f'input: "{nodeName}"',
                f'input: "{nodeName}/mul"')

    return content


FILE_NAME = sys.argv[1]
PBTXT_FILE_NAME = FILE_NAME.replace('.pb', '.pbtxt')

OUTPUT_FILTER = 1280
if len(sys.argv) > 2:
    OUTPUT_FILTER = int(sys.argv[2])

if not os.path.isfile(PBTXT_FILE_NAME):
    f = open(FILE_NAME, 'rb')
    GRAPH_DEF = tf.compat.v1.get_default_graph().as_graph_def(add_shapes=True)
    GRAPH_DEF.ParseFromString(f.read())
    f.close()

    tf.import_graph_def(GRAPH_DEF, name='')
    io.write_graph(GRAPH_DEF, '', PBTXT_FILE_NAME, as_text=True)
else:

    GRAPH_DEF = tf.get_default_graph().as_graph_def(add_shapes=True)

    FILE_CONTENT = pbtxt_processing(open(PBTXT_FILE_NAME, 'r').read())

    print('Content check OK, start merging...')

    text_format.Merge(FILE_CONTENT, GRAPH_DEF)
    io.write_graph(GRAPH_DEF,
                   os.path.dirname(FILE_NAME),
                   os.path.basename(FILE_NAME).split('.')[0] + '_patched.pb',
                   as_text=False)
    
    os.remove(PBTXT_FILE_NAME)
