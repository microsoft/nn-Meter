# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import yaml
import logging
logging = logging.getLogger("nn-Meter")


__user_config_folder__ = os.path.expanduser('~/.nn_meter/config')
__registry_cfg_filename__ = 'registry.yaml'

def list_backends_cli():
    from nn_meter.builder.backends import list_backends
    backends = list_backends()
    logging.keyinfo("Supported backends: ('*' indicates customized backends)")
    for name in backends:
        logging.result(f"[Backend] {name}")


def list_operators_cli():
    from nn_meter.builder.backend_meta.fusion_rule_tester.utils import list_operators
    operators = list_operators()
    logging.keyinfo("Supported operators: ('*' indicates customized operators)")
    for name in operators:
        logging.result(f"[Operator] {name}")


def list_kernels_cli():
    from nn_meter.builder.kernel_predictor_builder.data_sampler.utils import list_kernels
    kernels = list_kernels()
    logging.keyinfo("Supported kernels: ('*' indicates customized kernels)")
    for name in kernels:
        logging.result(f"[Kernel] {name}")


def list_special_testcases_cli():
    from nn_meter.builder.backend_meta.fusion_rule_tester.generate_testcase import list_testcases
    testcases = list_testcases()
    logging.keyinfo("Supported testcases: ('*' indicates customized test cases)")
    for name in testcases:
        logging.result(f"[TestCase] {name}")


def create_workspace_cli(args):
    """create a workspace folder and copy the corresponding config file to the workspace
    """
    from nn_meter.builder.config_manager import copy_to_workspace
    
    if args.tflite_workspace:
        backend_type = "tflite"
        workspace_path = args.tflite_workspace
        copy_to_workspace(backend_type, workspace_path)
    elif args.openvino_workspace:
        backend_type = "openvino"
        workspace_path = args.openvino_workspace
        # create openvino_env
        openvino_env = os.path.join(workspace_path, 'openvino_env')
        os.system(f"virtualenv {openvino_env}")
        os.system("source {openvino_env}/bin/activate")
        os.system("pip install -r docs/requirements/openvino_requirements.txt")
        os.system("deactivate")
        copy_to_workspace(backend_type, workspace_path)
    elif args.customized_workspace:
        backend_type = "customized"
        backend_name = args.backend
        workspace_path = args.customized_workspace
        if os.path.isfile(os.path.join(__user_config_folder__, __registry_cfg_filename__)):
            with open(os.path.join(__user_config_folder__, __registry_cfg_filename__), 'r') as fp:
                registry_modules = yaml.load(fp, yaml.FullLoader)
            try:
                backend_config = registry_modules["backends"][backend_name]["defaultConfigFile"]
                copy_to_workspace(backend_type, workspace_path, backend_config)
            except:
                raise ValueError(f"Create workspace failed. Please check the backend registration information.")
        else:
            raise ValueError(f"Create workspace failed. Please check the backend registration information.")
    else:
        logging.keyinfo('please run "nn-meter create --help" to see guidance.')
        return

    logging.keyinfo(f"Workspace {os.path.abspath(workspace_path)} for {backend_type} platform has been created. " \
        f"Users could edit experiment config in {os.path.join(os.path.abspath(workspace_path), 'configs/')}.")


def test_backend_connection_cli(args):
    from nn_meter.builder import builder_config
    from nn_meter.builder.backends import connect_backend
    if args.workspace and args.backend:
        builder_config.init(args.workspace)
        backend = connect_backend(args.backend)
        backend.test_connection()
    else:
        logging.keyinfo('please run "nn-meter connect --help" to see guidance.')
