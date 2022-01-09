# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


def get_op_is_two_inputs(op_name):
    if op_name in ["add", "concat"]:
        return True
    else:
        return False


def save_model(model, savepath):
    from tensorflow import keras 
    keras.models.save_model(model, savepath)
