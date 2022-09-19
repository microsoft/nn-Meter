import subprocess


def get_pyver(interpreter_path):
    script = '''
import sys
print(f'{sys.version_info.major}.{sys.version_info.minor}')
    '''
    version = subprocess.check_output(f'{interpreter_path} -c "{script}"', shell=True)
    return version.decode('utf-8').strip()
