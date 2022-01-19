# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import logging


def list_backends_cli():
    from nn_meter.builder.backends import list_backends
    backends = list_backends()
    logging.keyinfo("Supported backends:")
    for name in backends.keys():
        logging.result(f"[Backend] {name}")
    return


def list_operators_cli():
    pass


def list_kernels_cli():
    pass


def list_special_testcases_cli():
    pass


def create_workspace_cli(args):
    """create a workspace folder and copy the corresponding config file to the workspace
    """
    if args.customized_workspace:
        # backend_name, workspace_path, config_path = args.customized_workspace
        # os.makedirs(workspace_path, exist_ok=True)
        
        # from nn_meter.builder import copy_cusconfig_to_workspace
        # copy_cusconfig_to_workspace(workspace_path, config_path)
        
        # logging.keyinfo(f'Create workspace at {workspace_path}.')
        # return
        pass
    
    if args.tflite_workspace:
        backend_type = "tflite"
        workspace_path = args.tflite_workspace
    elif args.openvino_workspace:
        backend_type = "openvino"
        workspace_path = args.openvino_workspace
        # create openvino_env
        openvino_env = os.path.join(workspace_path, 'openvino_env')
        os.system(f"virtualenv {openvino_env}")
        os.system("source {openvino_env}/bin/activate")
        os.system("pip install -r docs/requirements/openvino_requirements.txt")
        os.system("deactivate")
    
    from nn_meter.builder.config_manager import copy_to_workspace
    copy_to_workspace(backend_type, workspace_path)
    logging.keyinfo(f"Workspace {os.path.abspath(workspace_path)} for {backend_type} platform has been created. " \
        f"Users could edit experiment config in {os.path.join(os.path.abspath(workspace_path), 'configs/')}.")


def test_backend_connection_cli(args):
    from nn_meter.builder import builder_config
    from nn_meter.builder.backends import connect_backend
    builder_config.init(args.workspace)
    backend = connect_backend(args.backend)
    backend.test_connection()
