# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# TODO: add it in registration
def get_op_is_two_inputs(op_name):
    if op_name in ["add", "concat"]:
        return True
    else:
        return False
