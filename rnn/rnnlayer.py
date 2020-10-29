from layer import Layer
import numpy as np

class RNNLayer(Layer):
    def __init__(self, hidden_size, bias=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.hidden = []
        self.self_weight = []
        self.input_weight = []
        self.bias_weight = []

        return None

    def init_params(self):
        # Init weights
        WEIGHT_LOWER_BOUND = -1
        WEIGHT_UPPER_BOUND = 1
        
        self.self_weight = np.random.uniform(
            low=WEIGHT_LOWER_BOUND, 
            high=WEIGHT_UPPER_BOUND, 
            size=(self.hidden_size, self.hidden_size))

        self.input_weight = np.random.uniform(
            low=WEIGHT_LOWER_BOUND, 
            high=WEIGHT_UPPER_BOUND, 
            size=(self.hidden_size, self.input_shape))

        self.bias_weight = np.random.uniform(
            low=WEIGHT_LOWER_BOUND, 
            high=WEIGHT_UPPER_BOUND, 
            size=(self.hidden_size))

        self.hidden = np.zeros(self.hidden_size)

        return None

    def run(self):
        self.output = np.zeros(self.hidden_size)

        # Calculate net output
        np.matmul(self.input_weight, self.input) 
        + np.matmul(self.self_weight, self.hidden) 
        + (self.bias * self.bias_weight)

        return None
        
    def calculate_error(self):
        return None
                
    def update_weight(self):
        return None


    

    