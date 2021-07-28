# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from .nn_meter import (
    nnMeter,
    load_latency_predictor,
    list_latency_predictors,
)
from .utils.utils import download_from_url
import logging
from functools import partial, partialmethod

logging.KEYINFO = 22
logging.addLevelName(logging.KEYINFO, 'KEYINFO')
logging.Logger.keyinfo = partialmethod(logging.Logger.log, logging.KEYINFO)
logging.keyinfo = partial(logging.log, logging.KEYINFO)

logging.RESULT = 25
logging.addLevelName(logging.RESULT, 'RESULT')
logging.Logger.result = partialmethod(logging.Logger.log, logging.RESULT)
logging.result = partial(logging.log, logging.RESULT)
