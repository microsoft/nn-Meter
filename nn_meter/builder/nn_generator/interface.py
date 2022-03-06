# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

class BaseOperator:
    def __init__(self, input_shape, config=None):
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
    def __init__(self, config):
        ''' base class for kernel block.
        
        @params
        
        - `input_shape`: defines the dimension of one model input shape without batch size. Generally, when the input shape is 3D, `input_shape`
            should be`[config["HW"], config["HW"], config["CIN"]]`, and when the input shape is 1D, `input_shape` should be`[config["CIN"]]`.

        - `input_tensor_shape`: a list defining all model inputs. In basic situation, `input_tensor_shape` should be `[input_shape]` if the kernel
            only has one input. If the kernel has more than one input, such as `add_relu` kernel, `input_tensor_shape` is `[input_shape, input_shape]`.
        '''
        self.config = config
        self.input_shape = None
        self.input_tensor_shape = None

    def get_model(self):
        ''' the implementation of the kernel model and return a instance of `keras.Model` of the kernel.
        '''
        pass
    
    def test_block(self):
        import os, shutil
        from typing import List
        model_path = "./temp_model"
        model = self.get_model()
        model_output = model(get_inputs_by_shapes(self.input_tensor_shape))
        
        # check model save and reload
        keras.models.save_model(model, model_path)
        restore_model = keras.models.load_model(model_path)
        if isinstance(model_output, List):
            output_shape = [mod.shape for mod in model_output]
            restore_output_shape = [mod.shape for mod in restore_model(get_inputs_by_shapes(self.input_tensor_shape))]
        else:
            output_shape = model_output.shape
            restore_output_shape = restore_model(get_inputs_by_shapes(self.input_tensor_shape)).shape
        assert output_shape == restore_output_shape
        shutil.rmtree(model_path)

        # check model convert to tflite
        converter = tf.lite.TFLiteConverter.from_keras_model(restore_model)
        tflite_model = converter.convert()
        open(model_path + '.tflite', 'wb').write(tflite_model)
        os.remove(model_path + '.tflite')
        logging.keyinfo("Testing block is success!")
