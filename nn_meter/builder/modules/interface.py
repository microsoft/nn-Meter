# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

class BaseOperator:
    def __init__(self, input_shape = None, config = None):
        ''' base class for operator.
        
        @params
        
        - input_shape: defines the dimension of one model input shape without batch size. Generally, when the input shape is 3D, `input_shape`
            should be `[config["HW"], config["HW"], config["CIN"]]` (for tensorflow model), or `config["CIN"], [config["HW"], config["HW"]]` (for
            torch model), and when the input shape is 1D, `input_shape` should be`[config["CIN"]]`.

        - config: a dict containing all configurations.
        '''
        self.input_shape = input_shape
        self.config = config

    def get_model(self):
        pass

    def get_output_shape(self):
        return self.input_shape

    def get_is_two_inputs(self):
        return False

    def test_operator():
        ''' for users to test the model when registration. Do not need to override by users.
        '''
        pass


class BaseBlock:
    def __init__(self, config, batch_size = 1):
        ''' base class for kernel block.
        
        @params
        
        - input_shape: defines the dimension of one model input shape without batch size. Generally, when the input shape is 3D, `input_shape`
            should be `[config["HW"], config["HW"], config["CIN"]]` (for tensorflow model), or `config["CIN"], [config["HW"], config["HW"]]` (for
            torch model), and when the input shape is 1D, `input_shape` should be`[config["CIN"]]`.

        - input_tensor_shape: a list defining all model inputs. In basic situation, `input_tensor_shape` should be `[input_shape]` if the kernel
            only has one input. If the kernel has more than one input, such as `addrelu` kernel, `input_tensor_shape` is `[input_shape, input_shape]`.

        - batch_size: the required batch size of the input data
        '''
        self.config = config
        self.input_shape = None
        self.input_tensor_shape = None
        self.batch_size = batch_size

    def get_model(self):
        ''' the implementation of the kernel model and return a instance of `tensorflow.keras.Model` or `torch.nn.Module` of the kernel.
        '''
        pass

    def save_model(self, save_path):
        pass

    def test_block(self):
        pass