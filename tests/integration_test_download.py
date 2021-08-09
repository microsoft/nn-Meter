import os
from nn_meter import download_from_url

if __name__ == "__main__":
    if not os.path.isdir('~/.nn_meter/cortexA76cpu_tflite21'):
        download_from_url('https://github.com/microsoft/nn-Meter/releases/download/v1.0-data/cortexA76cpu_tflite21.zip', '~/.nn_meter')
        print("complete download in ~/.nn_meter")
        
    else:
        print("found ~/.nn_meter/cortexA76cpu_tflite21, download is not needed")
    print('############### file list')
    for file in os.listdir('~/.nn_meter/cortexA76cpu_tflite21'):
        print(os.path.abspath(os.path.join('~/.nn_meter/cortexA76cpu_tflite21', file)))
    print('############### ls -R ~/.nn_meter')
    os.system('ls -R ~/.nn_meter')
    print('############### ls -R ' + os.path.abspath('~/.nn_meter'))
    os.system('ls -R ' + os.path.abspath('~/.nn_meter'))

