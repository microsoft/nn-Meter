from ..tflite_runner import TFLiteRunner


class TFLiteGPURunner(TFLiteRunner):
    use_gpu = True
