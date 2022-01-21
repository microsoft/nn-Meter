**1. Before register**

Run command to list all backend:
``` bash
nn-meter --list-backends
```
``` text
(nn-Meter) Supported backends: ('*' indicates customized backends)
(nn-Meter) [Backend] tflite_cpu
(nn-Meter) [Backend] tflite_gpu
(nn-Meter) [Backend] openvino_vpu
```

Run command to see registration guidance:
```bash
nn-meter register --h
```
```
usage: nn-meter register [-h]
                         [--predictor PREDICTOR | --backend BACKEND | --operator OPERATOR | --testcase TESTCASE | --kernel KERNEL]

optional arguments:
  -h, --help            show this help message and exit
  --predictor PREDICTOR
                        path to the meta file to register a customized
                        predictor
  --backend BACKEND     path to the meta file to register a customized backend
  --operator OPERATOR   path to the meta file to register a customized
                        operator
  --testcase TESTCASE   path to the meta file to register a customized
                        testcase
  --kernel KERNEL       path to the meta file to register a customized kernel
```

**2. Register a new backend**

Prepare customized backend (in `/data/jiahang/working/tftest/test_package_import/backend.py`):
``` python
from nn_meter.builder.backends import BaseBackend

class myBackend(BaseBackend):
    def __init__(self, config):
        pass

    def test_connection(self):
        """check the status of backend interface connection
        """
        print("Registration Success!")
```

Users should provide a folder as a package with a `__init__.py`, and containing all dependencies in the folder, such as `Parser`, `Runner` for customized backend. In this demo, the folder information is:

``` text
/data/jiahang/working/tftest/test_package_import/
├── __init__.py
├── backend.py
├── config.yaml
└── meta_backend.yaml
```

Meta file content:

``` yaml
builtinName: my_backend
packageLocation: /data/jiahang/working/tftest/test_package_import
classModule: backend
className: myBackend
defaultConfigFile: /data/jiahang/working/tftest/test_package_import/config.yaml
```

The meta file could be deleted after registration, as nn-Meter has copied the information in nn-Meter config. However, the customized backend code must be retained in the exactly same location as registered one. Otherwise will cause error when utilizing the register module.

Run register command in command line bash:

``` bash
nn-meter register --backend /data/jiahang/working/tftest/test_package_import/meta_backend.yaml
```
``` text
(nn-Meter) Successfully register backend my_backend
```

Registry information can be seen in `~/.nn_meter/config/registry.yaml`:

``` yaml
backends:
  my_backend:
    classModule: backend
    className: myBackend
    packageLocation: /data/jiahang/working/tftest/test_package_import
```

**3. Test registered backend**

Help of `nn-meter create`:
``` text
usage: nn-meter create [-h]
                       [--tflite-workspace TFLITE_WORKSPACE | --openvino-workspace OPENVINO_WORKSPACE | --customized-workspace CUSTOMIZED_WORKSPACE]
                       [--backend BACKEND]

optional arguments:
  -h, --help            show this help message and exit
  --tflite-workspace TFLITE_WORKSPACE
                        path to place a tflite workspace for nn-Meter builder
  --openvino-workspace OPENVINO_WORKSPACE
                        path to place a openvino workspace for nn-Meter
                        builder
  --customized-workspace CUSTOMIZED_WORKSPACE
                        path to place a customized workspace for nn-Meter
                        builder. A customized backend should be register first
                        (refer to `nn-meter register --h` for more help).
  --backend BACKEND     the backend name for registered backend
```

Create customized workspace according to the customized backend:

``` bash
nn-meter create --customized-workspace /data/jiahang/working/custom/ --backend my_backend
```
``` text
(nn-Meter) Workspace /data/jiahang/working/custom for customized platform has been created. Users could edit experiment config in /data/jiahang/working/custom/configs/.
```
Workspace structure:
``` text
/data/jiahang/working/custom/
└── configs
    ├── backend_config.yaml         # copy from customized config file
    ├── predictorbuild_config.yaml  # copy from nn-Meter package
    └── ruletest_config.yaml        # copy from nn-Meter package
```

Test the connection to the registered backend:

``` bash
nn-meter connect --backend my_backend --workspace /data/jiahang/working/tftest/
```
```
Registration Success!
```

List all backends:

``` bash
nn-meter --list-backends
```
```text
(nn-Meter) Supported backends: ('*' indicates customized backends)
(nn-Meter) [Backend] tflite_cpu
(nn-Meter) [Backend] tflite_gpu
(nn-Meter) [Backend] openvino_vpu
(nn-Meter) [Backend] * my_backend
```

When registering, nn-Meter will test whether the module can be imported first. If the module referring to meta file cannot be imported in registration:

``` bash
nn-meter register --backend /data/jiahang/working/tftest/test_package_import/meta_file.yaml 
```
``` text
Traceback (most recent call last):
  File "/home/jiahang/.conda/envs/py36-Jiahang/bin/nn-meter", line 8, in <module>
    sys.exit(nn_meter_cli())
  File "/home/jiahang/.conda/envs/py36-Jiahang/lib/python3.6/site-packages/nn_meter/utils/nn_meter_cli/interface.py", line 256, in nn_meter_cli
    args.func(args)
  File "/home/jiahang/.conda/envs/py36-Jiahang/lib/python3.6/site-packages/nn_meter/utils/nn_meter_cli/registry.py", line 47, in register_module_cli
    register_module("backends", args.backend)
  File "/home/jiahang/.conda/envs/py36-Jiahang/lib/python3.6/site-packages/nn_meter/utils/nn_meter_cli/registry.py", line 27, in register_module
    import_module(meta_data)
  File "/home/jiahang/.conda/envs/py36-Jiahang/lib/python3.6/site-packages/nn_meter/utils/nn_meter_cli/registry.py", line 17, in import_module
    module = importlib.import_module(module_path)   
  File "/home/jiahang/.conda/envs/py36-Jiahang/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
ModuleNotFoundError: No module named 'backend'
```

and the backend will not be registered to nn-Meter:

``` bash
nn-meter --list-backends
```
```
(nn-Meter) Supported backends: ('*' indicates customized backends)
(nn-Meter) [Backend] tflite_cpu
(nn-Meter) [Backend] tflite_gpu
(nn-Meter) [Backend] openvino_vpu
```

**Discussion**: should we just throw the error information or raise a logging information as "registration failed"?

If the file containing the backend code is moved after registration:

``` bash
nn-meter connect --backend my_backend --workspace /data/jiahang/working/tftest/
```
```
Traceback (most recent call last):
  File "/home/jiahang/.conda/envs/py36-Jiahang/bin/nn-meter", line 8, in <module>
    sys.exit(nn_meter_cli())
  File "/home/jiahang/.conda/envs/py36-Jiahang/lib/python3.6/site-packages/nn_meter/utils/nn_meter_cli/interface.py", line 256, in nn_meter_cli
    args.func(args)
  File "/home/jiahang/.conda/envs/py36-Jiahang/lib/python3.6/site-packages/nn_meter/utils/nn_meter_cli/builder.py", line 65, in test_backend_connection_cli
    backend = connect_backend(args.backend)
  File "/home/jiahang/.conda/envs/py36-Jiahang/lib/python3.6/site-packages/nn_meter/builder/backends/interface.py", line 180, in connect_backend
    backend_module = importlib.import_module(module)   
  File "/home/jiahang/.conda/envs/py36-Jiahang/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
ModuleNotFoundError: No module named 'backend'
```

**4. Unregister the backend**

Run command line:

``` bash
nn-meter unregister --backend my_backend
```
``` text
(nn-Meter) Successfully unregister my_backend.
```

After unregister the backend, users cannot test the connection to this backend:

``` bash
nn-meter connect --backend my_backend --workspace /data/jiahang/working/tftest/
```
``` text
Traceback (most recent call last):
  File "/home/jiahang/.conda/envs/py36-Jiahang/bin/nn-meter", line 8, in <module>
    sys.exit(nn_meter_cli())
  File "/home/jiahang/.conda/envs/py36-Jiahang/lib/python3.6/site-packages/nn_meter/utils/nn_meter_cli/interface.py", line 256, in nn_meter_cli
    args.func(args)
  File "/home/jiahang/.conda/envs/py36-Jiahang/lib/python3.6/site-packages/nn_meter/utils/nn_meter_cli/builder.py", line 65, in test_backend_connection_cli
    backend = connect_backend(args.backend)
  File "/home/jiahang/.conda/envs/py36-Jiahang/lib/python3.6/site-packages/nn_meter/builder/backends/interface.py", line 176, in connect_backend
    raise ValueError(f"Unsupported backend name: {backend_name}. Please register the backend first.")
ValueError: Unsupported backend name: my_backend. Please register the backend first.
```

Finally, list all supporting backends by now:

``` bash
nn-meter --list-backends
```
``` text
(nn-Meter) Supported backends: ('*' indicates customized backends)
(nn-Meter) [Backend] tflite_cpu
(nn-Meter) [Backend] tflite_gpu
(nn-Meter) [Backend] openvino_vpu
```