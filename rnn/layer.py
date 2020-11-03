import abc

class Layer(metaclass=abc.ABCMeta):
    def __init__(self, input_shape=None):
        self.input = []
        self.output = []
        self.input_shape = input_shape
        
        # Error calculation related variables
        self.error_calc_output = [] # for output layer
        self.prev_error = [] # error from backprop previous layer
        self.passed_error = [] # error passed to backprop next layer
        self.error = [] # error calculation result for this layer

        return None

    @abc.abstractmethod
    def init_params(self):
        pass
        
    @abc.abstractmethod
    def run(self):
        pass
        
    @abc.abstractmethod
    def calculate_error(self):
        pass
                
    @abc.abstractmethod
    def update_weight(self):
        pass
    
    def calculate_delta_output(self):
        return None # kalo output layer


    