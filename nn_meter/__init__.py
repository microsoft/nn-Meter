# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

try:
    import pkg_resources  # part of setuptools
    __version__ = pkg_resources.require("nn-meter")[0].version
except ModuleNotFoundError:
    __version__ = 'UNKNOWN'

from functools import partial, partialmethod

from .predictor import (
    nnMeterPredictor,
    load_latency_predictor,
    list_latency_predictors,
    latency_metrics
)
from .ir_converter import (
    model_file_to_graph,
    model_to_graph
)
from .utils import (
    create_user_configs,
    change_user_data_folder,
    download_from_url
)
from .dataset import bench_dataset

# setup logging
import sys
import logging
logging.KEYINFO = 22
logging.addLevelName(logging.KEYINFO, 'KEYINFO')
logging.Logger.keyinfo = partialmethod(logging.Logger.log, logging.KEYINFO)
logging.keyinfo = partial(logging.log, logging.KEYINFO)

logging.RESULT = 25
logging.addLevelName(logging.RESULT, 'RESULT')
logging.Logger.result = partialmethod(logging.Logger.log, logging.RESULT)
logging.result = partial(logging.log, logging.RESULT)

from logging import Formatter, StreamHandler
handler = StreamHandler(sys.stdout)
handler.setFormatter(Formatter("(nn-Meter) %(message)s"))
logger = logging.getLogger("nn-Meter")
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False