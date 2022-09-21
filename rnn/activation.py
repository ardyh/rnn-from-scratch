from .layer import Layer 
import numpy as np

class Activation(Layer):
    def __init__(self, function_name="softmax", class_num=None):
        super().__init__()
        self.function_name = function_name
        self.class_num = class_num

    def init_params(self):
        return None

    def sigmoid(self, net):
        return 1. / (1. + np.exp(-net))

    def relu(self, net):
        return 0 if (net < 0) else net

    def tanh(self, net):
        return np.tanh(net)

    def exp(self, net):
        return np.exp(-net)

    def softmax(self, index, total):
        return index / total
    
    def run(self):
        function_name = self.function_name
        
        # activation function defaults to relu
        if (function_name == "sigmoid"):
            v_activation = np.vectorize(self.sigmoid)
        elif (function_name == "relu"):
            v_activation = np.vectorize(self.relu)
        elif (function_name == "tanh"):
            v_activation = np.vectorize(self.tanh)
        elif (function_name == "softmax"):
            v_activation = np.vectorize(self.exp)
        else:
            raise Exception("Invalid activation function name")

        if (function_name != "softmax"):
            self.output = v_activation(self.input)
        else:
            temp = v_activation(self.input)
            total = np.sum(temp)

            v_activation = np.vectorize(self.softmax)
            self.output = v_activation(temp, total)
        
        return None
