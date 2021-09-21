from ..tflite_runner import TFLiteRunner


class TFLiteCPURunner(TFLiteRunner):
    use_gpu = False
