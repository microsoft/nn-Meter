import logging
from nn_meter.builder.backends import BaseBackend, BaseParser, BaseProfiler

class MyParser(BaseParser): ...

class MyProfiler(BaseProfiler): ...

class MyBackend(BaseBackend):
    parser_class = MyParser
    profiler_class = MyProfiler

    def __init__(self, config):
        pass

    def test_connection(self):
        """check the status of backend interface connection
        """
        ...
        logging.keyinfo("hello backend !")
