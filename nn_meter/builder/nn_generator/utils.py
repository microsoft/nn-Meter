# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


def get_op_is_two_inputs(op_name):
    if op_name in ["conv", "dwconv", "convtrans", "avgpool", "se", "dense", "relu", "hswish", "reshape"]:
        return False
    elif op_name in ["add", "concat"]:
        return True
    else:
        raise ValueError(f"Unsupported operator name: {op_name} in rule-tester.")


def save_testcase(model, savepath):
    from tensorflow import keras 
    keras.models.save_model(model, savepath)
