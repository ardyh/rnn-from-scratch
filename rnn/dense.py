from layer import Layer 
import numpy as np

class Dense(Layer):
    def __init__(self, class_num=3, bias=1, momentum=0.01, learning_rate=0.05):
        super().__init__()

        self.class_num = class_num
        self.weights = []
        self.bias = bias
        self.momentum = momentum
        self.learning_rate = learning_rate
        # Error calculation, backprop
        self.delta_weight = []

        # to be deleted
        # self.prev_error = []
        # self.passed_error = []
        # self.error = []
        # self.input = []
        # self.output = []

    def init_params(self):
        class_num = self.class_num

        # Validation
        if class_num <= 0 :
            raise Exception("Invalid class number")

        # Init weight
        input_size = 1

        return None

    def reset_delta_weight(self):
        self.delta_weight = np.zeros(self.weights.shape)

    def calculate_error(self):
        # Error to be passed to previous layer
        # dNet/dRelu = weight
        # dError/dRelu
        self.passed_error = np.matmul(
            self.prev_error.reshape(1, self.prev_error.size),
            self.weights
        )[:, :-1] # excluding bias

        # calculate derived error for layer weight
        # dNet/dWeight = input
        # dError/dWeight
        self.error = np.matmul(
            self.prev_error.reshape(self.prev_error.size, 1),
            self.input.reshape(1, self.input.size),
        )

        # Calculate delta_weight with momentum
        # Assumption: delta_weight(n) = error + alpha * delta_weight(n-1)
        self.delta_weight = self.error + self.momentum * self.delta_weight

    def update_weight(self):
        # Assumption: weight(n) = weight(n-1) - lr * delta_weight(n-1)
        self.weights = self.weights - self.learning_rate * self.delta_weight
        self.reset_delta_weight()

    def run(self):
        # Init variables
        self.input = np.append(self.input, self.bias) # Append bias to input. Assumption: input is already flattened
        
        # Init weight if first forwardprop
        if self.weights == []:
            self.weights = np.random.uniform(-1, 1, (self.class_num, self.input.size))
            self.reset_delta_weight()

        class_num = self.class_num
        input_size = self.input.size
        self.output = np.zeros(class_num)

        # Calculate net output
        for c in range(class_num):
            for w in range(input_size):
                self.output[c] += self.input[w] * self.weights[c][w] 

        return None

    def backward(self):
        pass
