# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
from packaging import version
logging = logging.getLogger("nn-Meter")


def try_import_onnx(require_version = ["1.10.0", "1.9.0"]):
    if isinstance(require_version, str):
        require_version = [require_version]
    try:
        import onnx
        if version.parse(onnx.__version__).release[:2] not in [version.parse(v).release[:2] for v in require_version]:
            logging.warning(f'onnx=={onnx.__version__} is not well tested now, well tested version: onnx=={", ".join(require_version)}' )
        return onnx
    except ImportError:
        logging.error(f'You have not install the onnx package, please install onnx=={require_version[0]} and try again.')


def try_import_torch(require_version = ["1.10.0", "1.9.0", "1.7.1"]):
    if isinstance(require_version, str):
        require_version = [require_version]
    try:
        import torch
        if version.parse(torch.__version__).release[:2] not in [version.parse(v).release[:2] for v in require_version]:
            logging.warning(f'torch=={torch.__version__} is not well tested now, well tested version: torch=={", ".join(require_version)}' )
        return torch
    except ImportError:
        logging.error(f'You have not install the torch package, please install torch=={require_version[0]} and try again.')


def try_import_tensorflow(require_version = ["2.7.0", "2.6.0"]):
    if isinstance(require_version, str):
        require_version = [require_version]
    try:
        import tensorflow
        if version.parse(tensorflow.__version__).release[:2] not in [version.parse(v).release[:2] for v in require_version]:
            logging.warning(f'tensorflow=={tensorflow.__version__} is not well tested now, well tested version: tensorflow=={", ".join(require_version)}' )
        return tensorflow
    except ImportError:
        logging.error(f'You have not install the tensorflow package, please install tensorflow=={require_version[0]} and try again.')


def try_import_nni(require_version = ["2.7", "2.6", "2.5", "2.4"]):
    if isinstance(require_version, str):
        require_version = [require_version]
    try:
        import nni
        if version.parse(nni.__version__).release not in [version.parse(v).release for v in require_version]:
            logging.warning(f'nni=={nni.__version__} is not well tested now, well tested version: nni=={", ".join(require_version)}' )
        return nni
    except ImportError:
        logging.error(f'You have not install the nni package, please install nni=={require_version[0]} and try again.')


def try_import_torchvision_models():
    try:
        import torchvision
        return torchvision.models
    except ImportError:
        logging.error(f'You have not install the torchvision package, please install torchvision and try again.')


def try_import_onnxsim():
    try:
        from onnxsim import simplify
        return simplify
    except ImportError:
        logging.error(f'You have not install the onnx-simplifier package, please install onnx-simplifier and try again.')


def try_import_dgl():
    try:
        import dgl
        return dgl
    except ImportError:
        logging.error(f'You have not install the dgl package, please install dgl and try again.')

    