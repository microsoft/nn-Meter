# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from .rule_tester.runner import RuleTester
from nn_meter.utils.utils import try_import_tensorflow

try_import_tensorflow()
