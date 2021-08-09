from nn_meter import download_from_url

if __name__ == "__main__":
    download_from_url('https://github.com/microsoft/nn-Meter/releases/download/v1.0-data/cortexA76cpu_tflite21.zip', '~/.nn_meter')
    print("complete download")
