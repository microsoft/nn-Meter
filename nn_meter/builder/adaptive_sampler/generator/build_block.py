# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from .networks.ops.primitive_ops import *
from .networks.block_utils import *

def build_block(block_type, cfg):
      """ returns: input_tensors, output_tensors, configuration_key, and graphname, they are for saving tensorflow v1.x models
      """
      if block_type == 'conv-bn-relu':
          hw_in = cfg['HW']
          cin = cfg['CIN']
          cout = cfg['COUT']
          kernel_size = cfg['KERNEL_SIZE']
          stride = cfg['STRIDE']
          configs = [hw_in, cin, cout, kernel_size, stride]
          graphname = convbnrelu.__name__
          inputs = generate_input_tensor([[1, hw_in, hw_in, cin]])
          input_tensor = inputs[0]
          out = convbnrelu(input_tensor, kernel_size, cin, cout, stride)
          #outputs = [res_out]
          return input_tensor, out, '_'.join([str(x) for x in configs]), graphname

      if block_type == 'conv':
          hw_in = cfg['HW']
          cin = cfg['CIN']
          cout = cfg['COUT']
          kernel_size = cfg['KERNEL_SIZE']
          stride = cfg['STRIDE']
          configs = [hw_in, cin, cout, kernel_size, stride]
          graphname = conv.__name__
          inputs = generate_input_tensor([[1, hw_in, hw_in, cin]])
          input_tensor = inputs[0]
          out = conv(input_tensor, kernel_size, cin, cout, stride)
          #outputs = [res_out]
          return input_tensor, out, '_'.join([str(x) for x in configs]), graphname

      if block_type == 'conv-bn-relu-maxpool':
          hw_in = cfg['HW']
          cin = cfg['CIN']
          cout = cfg['COUT']
          kernel_size = cfg['KERNEL_SIZE']
          stride = cfg['STRIDE']
          configs = [hw_in, cin, cout, kernel_size, stride, 2, 2]
          graphname = convbnrelumaxpool.__name__
          inputs = generate_input_tensor([[1, hw_in, hw_in, cin]])
          input_tensor = inputs[0]
          out = convbnrelumaxpool(input_tensor, kernel_size, cin, cout, stride, 2, 2)
          #outputs = [res_out]
          return input_tensor, out, '_'.join([str(x) for x in configs]), graphname

      if block_type == 'conv-bn-hswish':
          hw_in = cfg['HW']
          cin = cfg['CIN']
          cout = cfg['COUT']
          kernel_size = cfg['KERNEL_SIZE']
          stride = cfg['STRIDE']
          configs = [hw_in, cin, cout, kernel_size, stride]
          graphname = convbnhswish.__name__
          inputs = generate_input_tensor([[1, hw_in, hw_in, cin]])
          input_tensor = inputs[0]
          out = convbnhswish(input_tensor, kernel_size, cin, cout, stride)
          #outputs = [res_out]
          return input_tensor, out, '_'.join([str(x) for x in configs]), graphname

      if block_type == 'dwconv-bn-relu':
          hw_in = cfg['HW']
          cin = cfg['CIN']
          kernel_size = cfg['KERNEL_SIZE']
          stride = cfg['STRIDE']
          configs = [hw_in, cin, kernel_size, stride]
          graphname = dwconvbnrelu.__name__
          inputs = generate_input_tensor([[1, hw_in, hw_in, cin]])
          input_tensor = inputs[0]
          out = dwconvbnrelu(input_tensor, kernel_size, cin, cin, stride)
          #outputs = [res_out]
          return input_tensor, out, '_'.join([str(x) for x in configs]), graphname

      if block_type == 'dwconv':
          hw_in = cfg['HW']
          cin = cfg['CIN']
          kernel_size = cfg['KERNEL_SIZE']
          stride = cfg['STRIDE']
          configs = [hw_in, cin, kernel_size, stride]
          graphname = dwconv.__name__
          inputs = generate_input_tensor([[1, hw_in, hw_in, cin]])
          input_tensor = inputs[0]
          out = dwconv(input_tensor, kernel_size, cin, cin, stride)
          #outputs = [res_out]
          return input_tensor, out, '_'.join([str(x) for x in configs]), graphname

      if block_type == 'dwconv-bn-hswish':
          hw_in = cfg['HW']
          cin = cfg['CIN']
          kernel_size = cfg['KERNEL_SIZE']
          stride = cfg['STRIDE']
          configs = [hw_in, cin, kernel_size, stride]
          graphname = dwconvbnhswish.__name__
          inputs = generate_input_tensor([[1, hw_in, hw_in, cin]])
          input_tensor = inputs[0]
          out = dwconvbnhswish(input_tensor, kernel_size, cin, cin, stride)
          #outputs = [res_out]
          return input_tensor, out, '_'.join([str(x) for x in configs]), graphname
      
      if block_type == 'hswish':
          hw_in = cfg['HW']
          cin = cfg['CIN']
          configs = [hw_in, cin]
          graphname = hswish.__name__
          inputs = generate_input_tensor([[1, hw_in, hw_in, cin]])
          input_tensor = inputs[0]
          out = hswish(input_tensor)
          #outputs = [res_out]
          return input_tensor, out, '_'.join([str(x) for x in configs]), graphname

      if block_type == 'se':
          hw_in = cfg['HW']
          cin = cfg['CIN']
          graphname = se.__name__
          inputs = generate_input_tensor([[1, hw_in, hw_in, cin]])
          input_tensor = inputs[0]
          out = se(input_tensor, cin)
          configs = [hw_in, cin, 4]
          #outputs = [res_out]
          return input_tensor, out, '_'.join([str(x) for x in configs]), graphname

      if block_type == 'fc':
            cin = cfg['CIN']
            cout = cfg['COUT']
            graphname = fc.__name__ 
            inputs = generate_input_tensor([[1, cin]])
            input_tensor = inputs[0]
            out = fc(input_tensor, cout)
           # print('out', out.shape)
            configs = [cin, cout]
            return input_tensor, out, '_'.join([str(x) for x in configs]), graphname

      if block_type == 'maxpool':
            cin = cfg['CIN']
            ks = cfg['KERNEL_SIZE']
            stride = cfg['STRIDE']
            hw = cfg['HW']
            configs = [hw, cin, ks, stride]
            inputs = generate_input_tensor([[1, hw, hw, cin]])
            input_tensor = inputs[0]
            graphname = max_pool.__name__ 
            out = max_pool(input_tensor, ks, stride)
            return input_tensor, out, '_'.join([str(x) for x in configs]), graphname

      if block_type == 'avgpool':
            cin = cfg['CIN']
            ks = cfg['KERNEL_SIZE']
            stride = cfg['STRIDE']
            hw = cfg['HW']
            configs = [hw, cin, ks, stride]
            inputs = generate_input_tensor([[1, hw, hw, cin]])
            input_tensor = inputs[0]
            graphname = avgpool.__name__ 
            out = avgpool(input_tensor, ks, stride)
            return input_tensor, out, '_'.join([str(x) for x in configs]), graphname

      if block_type == 'global-avgpool':
            cin = cfg['CIN']
            hw = cfg['HW']
            configs = [hw, cin]
            inputs = generate_input_tensor([[1, hw, hw, cin]])
            input_tensor = inputs[0]
            graphname = global_avgpool.__name__
            out = global_avgpool(input_tensor)
            return input_tensor, out, '_'.join([str(x) for x in configs]), graphname

      if block_type == 'split':
            cin = cfg['CIN']
            hw = cfg['HW']
            inputs = generate_input_tensor([[1, hw, hw, cin]])
            input_tensor = inputs[0]
            graphname = split.__name__
            out1, out2 = split(input_tensor)
            co = [cin, hw, 2]
            return input_tensor, [out1, out2], '_'.join([str(x) for x in co]), graphname

      if block_type == 'channel-shuffle':
            cin = cfg['CIN']
            hw = cfg['HW']
            inputs = generate_input_tensor([[1, hw, hw, cin]])
            input_tensor = inputs[0]
            graphname = channelshuffle.__name__
            out = channelshuffle(input_tensor)
            co = [cin, hw, 2]
            return input_tensor, out, '_'.join([str(x) for x in co]), graphname

      if block_type == 'bnrelu':
            cin = cfg['CIN']
            hw = cfg['HW']
            inputs = generate_input_tensor([[1, hw, hw, cin]])
            input_tensor = inputs[0]
            graphname = bnrelu.__name__
            out = bnrelu(input_tensor)
            co = [cin, hw]
            return input_tensor, out, '_'.join([str(x) for x in co]), graphname

      if block_type == 'concat':
            cins = cfg['CINS']
            hw = cfg['HW']
            inputts = []
            print('here')
            for c in cins:
                  inputs = generate_input_tensor([[1, hw, hw, c]])
                  input_tensor = inputs[0]
                  inputts.append(input_tensor)
            graphname = concats.__name__
            out = concats(inputts)
            print(out)
            co = [hw]+cins
            return inputts, out, '_'.join([str(x) for x in co]), graphname

      if block_type == 'concat-pad':
            cins = cfg['CINS']
            hw = cfg['HW']
            inputts = []
            inputs = generate_input_tensor([[1, hw, hw, 3]])
            input_tensor = inputs[0]
            graphname = concats_pad.__name__
            out = concats_pad(input_tensor, hw, cins)
            co = [hw]+cins
            return input_tensor, out, '_'.join([str(x) for x in co]), graphname

      if block_type == 'addrelu':
            cin = cfg['CIN']
            hw = cfg['HW']
            inputts = []
            for index in range(2):
                  inputs = generate_input_tensor([[1, hw, hw, cin]])
                  input_tensor = inputs[0]
                  inputts.append(input_tensor)
            graphname = addrelu.__name__
            out = addrelu(inputts)
            return input_tensor, out, '_'.join([str(x) for x in [hw, cin, cin]]), graphname

      if block_type == 'add':
            cin = cfg['CIN']
            hw = cfg['HW']
            inputts = []
            print('here')
            for index in range(2):
                  inputs = generate_input_tensor([[1, hw, hw, cin]])
                  input_tensor = inputs[0]
                  inputts.append(input_tensor)
            graphname = add.__name__
            out = add(inputts)
            print(out)
            co = [hw, cin]
            return inputts, out, '_'.join([str(x) for x in co]), graphname

      if block_type == 'bn':
            cin = cfg['CIN']
            hw = cfg['HW']
            inputs = generate_input_tensor([[1, hw, hw, cin]])
            input_tensor = inputs[0]
            graphname = bn.__name__
            out = bn(input_tensor)
            co = [cin, hw]
            return input_tensor, out, '_'.join([str(x) for x in co]), graphname

      if block_type == 'relu':
            cin = cfg['CIN']
            hw = cfg['HW']
            inputs = generate_input_tensor([[1, hw, hw, cin]])
            input_tensor = inputs[0]
            graphname = relu.__name__
            out = relu(input_tensor)
            co = [cin, hw]
            return input_tensor, out, '_'.join([str(x) for x in co]), graphname
