import os
from nn_meter import download_from_url

if __name__ == "__main__":
    if os.path.isdir('~/.nn_meter/cortexA76cpu_tflite21'):
        download_from_url('https://github.com/microsoft/nn-Meter/releases/download/v1.0-data/cortexA76cpu_tflite21.zip', '~/.nn_meter')
        print("complete download in ~/.nn_meter", )
    else:
        print("found ~/.nn_meter/cortexA76cpu_tflite21, download is not needed")