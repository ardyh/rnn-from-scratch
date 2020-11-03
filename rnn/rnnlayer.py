from .layer import Layer
import numpy as np

class RNNLayer(Layer):
    def __init__(self, units, bias=1, input_shape=None):
        super().__init__(input_shape)
        
        self.units = units
        self.bias = bias
        self.self_weight = []
        self.input_weight = []
        self.bias_weight = []

        self.init_params()

        return None

    def init_params(self):
        # Init weights
        WEIGHT_LOWER_BOUND = -1
        WEIGHT_UPPER_BOUND = 1
        
        self.self_weight = np.random.uniform(
            low=WEIGHT_LOWER_BOUND, 
            high=WEIGHT_UPPER_BOUND, 
            size=(self.units, self.units))

        self.input_weight = np.random.uniform(
            low=WEIGHT_LOWER_BOUND, 
            high=WEIGHT_UPPER_BOUND, 
            size=(self.units, self.input_shape))

        self.bias_weight = np.random.uniform(
            low=WEIGHT_LOWER_BOUND, 
            high=WEIGHT_UPPER_BOUND, 
            size=(self.units))

        self.output = np.zeros(self.units)

        return None

    def run(self):
        self.output = np.zeros(self.units)

        # Calculate net output
        self.output = np.matmul(self.input_weight, self.input) 
        + np.matmul(self.self_weight, self.output) 
        + (self.bias * self.bias_weight)

        return self.output
        
    def calculate_error(self):
        return None
                
    def update_weight(self):
        return None


    

    