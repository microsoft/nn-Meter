# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import pickle
import os

# multiple similar kernels share one kernel predictor, since the latency
# difference is negligible


def get_kernel_name(optype):
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
