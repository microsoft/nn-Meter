class TestCase:
    '''
    testcase[op] = {
                    'model': model,
                    'shapes': shapes
                }
    '''
    def __init__(self):
        self.data = {}
        self.after_run = False # check whether the test cases are run
        self.after_detect = False # check whether the test cases are detected
    
    def add(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data[key]

    def add_latency(self, key, latency):
        if self.with_latency == False:
            self.with_latency = True
    
    def add_rule():
        pass

    def __repr__(self) -> str:
        pass
    
    def _dump(self):
        return {name: graph._dump() for name, graph in self.graphs.items()}
