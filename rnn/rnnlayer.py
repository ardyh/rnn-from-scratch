from .layer import Layer
import numpy as np

class RNNLayer(Layer):
    def __init__(self, units, bias=1, init_weight_value=0, init_weight_random=True, input_shape=None):
        super().__init__(input_shape)
        
        self.units = units # number of hidden units
        self.bias = bias # single-value number, value of bias
        self.self_weight = [] # weight for hidden unit
        self.input_weight = [] # weight for input
        self.bias_weight = [] # weight for bias

        # Init weights
        self.set_initial_weight(init_weight_value, init_weight_random)

        return None

    # initialize layer parameters
    def init_params(self):
        # Init output shape
        self.output = np.zeros(self.units)

        return None

    # Initialize weights
    def set_initial_weight(self, value, random=False):
        WEIGHT_LOWER_BOUND = -1
        WEIGHT_UPPER_BOUND = 1

        self.matriks_u = np.random.uniform(
            low=WEIGHT_LOWER_BOUND,
            high=WEIGHT_UPPER_BOUND,
            size=(self.hidden_size, self.input_shape.shape[0])
        )

        self.matriks_w = np.random.uniform(
            low=WEIGHT_LOWER_BOUND,
            high=WEIGHT_UPPER_BOUND,
            size=(self.hidden_size, self.hidden_size)
        )

        self.matriks_v = np.random.uniform(
            low=WEIGHT_LOWER_BOUND,
            high=WEIGHT_UPPER_BOUND,
            size=(self.output_size, self.hidden_size)
        )

        self.bias_xh = np.zeros(self.hidden_size, 1)
        self.bias_hy = np.zeros(self.output_size,1)

        self.previous_ht = np.zeros(self.hidden_size, 1)
        self.previous_yt = np.zeros(self.output_size, 1)
        
        if(random):
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
        else:
            self.self_weight = np.full((self.units, self.units), value)
            self.input_weight = np.full((self.units, self.input_shape), value)
            self.bias_weight = np.full((self.units), value)

        return None

    # Reset hidden unit values (output)
    def reset_output(self):
        self.output = np.zeros(self.units)

    # Execute forward propagation
    def run(self):
        uv = np.matmul(self.input_weight, self.input) # multiply input_weight and input
        wh = np.matmul(self.self_weight, self.output) # multiply hidden_weight and previous hidden output
        b = self.bias * self.bias_weight # multiply bias and bias weight

        # Calculate net output
        self.output = uv + wh + b

        print('rnn input_weight\n', self.input_weight)
        print('rnn self_weight\n', self.self_weight)
        print('rnn bias_weight\n', self.bias_weight)
        print()
        print("uv\n", uv)
        print("wh\n", wh)
        print("b\n", b)
        print()
        print('rnn output :', self.output)

        # activation function
        self.output = np.tanh(self.output)
        print('rnn tanh output :', self.output)
        return self.output
