# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
package_location = os.path.join(BASE_DIR, "register_module")

def register_by_meta(module_type, register_meta):
    with open('meta_file.yaml', 'w') as fp:
        yaml.dump(register_meta, fp)
    os.system(f"nn-meter register --{module_type} meta_file.yaml")
    os.remove("meta_file.yaml")
    if module_type in ["operator", "testcase"]:
        implement = register_meta['implement']
    else:
        implement = ""
    os.system(f"nn-meter unregister --{module_type} {register_meta['builtin_name']} {implement}")


# test register backend
backend_meta = {
    "builtin_name": "my_backend",
    "package_location": package_location,
    "class_module": "backend",
    "class_name": "MyBackend",
    "defaultConfigFile": None
}
register_by_meta("backend", backend_meta)


# test register kernel
kernel_meta = {
    "builtin_name": "mykernel",
    "implement": "tensorflow",
    "package_location": package_location,
    "class_module": "kernel",
    "class_name": "MyKernel",
    "sampler_module": "kernel",
    "sampler_name": "MySampler",
    "parser_module": "kernel",
    "parser_name": "MyParser"
}
register_by_meta("kernel", kernel_meta)


# register operator
operator_meta = {
    "builtin_name": "myop",
    "implement": "tensorflow",
    "package_location": package_location,
    "class_module": "operators",
    "class_name": "MyOp"
}
register_by_meta("operator", operator_meta)


# register testcase
testcase_meta = {
    "builtin_name": "MyTC",
    "implement": "tensorflow",
    "package_location": package_location,
    "class_module": "testcase",
    "class_name": "MyTestCase"
}
register_by_meta("testcase", testcase_meta)
