from .layer import Layer 
import numpy as np

class Dense(Layer):
    def __init__(self, units, bias=1):
        super().__init__()

        self.units = units
        self.weights = []
        self.bias = bias

    # Initialize layer parameters
    def init_params(self):
        units = self.units

        # Validation
        if units <= 0 :
            raise Exception("num unit must be >0")

        # Init weight
        input_size = 1

        return None

    # Reset hidden unit values (output)
    def reset_output(self):
        self.output = np.zeros(self.units)

    # Execute forward propagation
    def run(self):
        # Init variables
        self.input = np.append(self.input, self.bias) # Append bias to input. Assumption: input is already flattened
        
        # Init weight if first forwardprop
        if self.weights == []:
            self.weights = np.random.uniform(-1, 1, (self.units, self.input.size))
            # self.reset_delta_weight()

        units = self.units
        input_size = self.input.size
        self.output = np.zeros(units)

        # Calculate net output
        for c in range(units):
            for w in range(input_size):
                self.output[c] += self.input[w] * self.weights[c][w] 

        # Activation function
        exp_output = np.exp(self.output)

        for i in range(self.units):
            self.output[i] = exp_output[i] / np.sum(exp_output)

        print('dense output : ',self.output)
        return None
