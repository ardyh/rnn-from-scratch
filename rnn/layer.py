import abc

class Layer(metaclass=abc.ABCMeta):
    def __init__(self, input_shape=None):
        self.input = []
        self.output = []
        self.input_shape = input_shape

        return None

    @abc.abstractmethod
    def init_params(self):
        pass
        
    @abc.abstractmethod
    def run(self):
        pass
