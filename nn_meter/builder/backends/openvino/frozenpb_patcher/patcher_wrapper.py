# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import subprocess


def patch_frozenpb(graph_path, interpreter_path):
    """ 
    Patch a frozen pb file to make it compatible with Movidius VPU and then return the path to the patched pb file.
    @params:

    graph_path: Path to the frozen pb file.

    interpreter_path:  the path of python interpreter
    """
    scripts_dir = os.path.abspath(os.path.dirname(__file__))
    subprocess.run(
        f'{interpreter_path} {os.path.join(scripts_dir, "frozenpb_patcher.py")} {graph_path}',
        shell=True
    )
    return os.path.splitext(graph_path)[0] + '_patched.pb'
