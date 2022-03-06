# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import torch
import torch.nn as nn
from .operators import *
from ..interface import BaseBlock
logging = logging.getLogger("nn-Meter")
