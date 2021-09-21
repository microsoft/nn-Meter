import subprocess
import os


def patch_frozen_pb(graph_path, interpreter_path):
    scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts')
    subprocess.run(
        f'{interpreter_path} {os.path.join(scripts_dir, "frozen_pb_patcher.py")} {graph_path}',
        shell=True
    )
    return os.path.splitext(graph_path)[0] + '_patched.pb'
