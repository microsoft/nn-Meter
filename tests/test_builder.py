import warnings
from silence_tensorflow import silence_tensorflow
warnings.filterwarnings('ignore')
silence_tensorflow() 

from nn_meter.builder.backends import connect_backend
from nn_meter.builder import create_testcases, run_testcases, detect_fusionrule

workspace_path = "" # text the path to the workspace folder. refer to ./backend.md for further information.


from nn_meter.builder import builder_config
builder_config.init(workspace_path)

# initialize backend
backend = connect_backend(backend='tflite_cpu')

# generate testcases
origin_testcases = create_testcases()

# run testcases and collect profiling results
profiled_testcases = run_testcases(backend, origin_testcases)

# determine fusion rules from profiling results
detected_testcases = detect_fusionrule(profiled_testcases)
