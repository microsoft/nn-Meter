# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

def get_kernel_name(optype):
    """
    for many similar kernels, we use one kernel predictor since their latency difference is negligible,
    return the kernel name via the optype

    """
    if "conv" in optype and "dwconv" not in optype:
        optype = "conv-bn-relu"
    if "dwconv" in optype:
        optype = "dwconv-bn-relu"
    if optype == "fc-relu":
        optype = "fc"
    if optype == "max-pool":
        optype = "maxpool"
    if optype == "avg-pool":
        optype = "avgpool"
    if optype in ["global-pool", "gap"]:
        optype = "global-avgpool"
    if optype == "channel_shuffle":
        optype = "channelshuffle"
    if optype in ["bn-relu"]:
        optype = "bnrelu"
    if optype in ["add-relu"]:
        optype = "addrelu"

    if optype in ["SE", "SE-relu", "se", "se-relu"]:
        optype = "se"

    return optype
