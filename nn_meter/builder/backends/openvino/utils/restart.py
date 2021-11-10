# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import time


def restart(ser):
    ser.write(b'all_toggle\n')
    time.sleep(0.5)
