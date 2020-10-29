from layer import Layer 
import numpy as np

class Activation(Layer):
    def __init__(self, function_name="relu", class_num=None):
        super().__init__()
        self.function_name = function_name
        self.class_num = class_num
        
        # Error calculation, backprop
        
        # to be deleted
        # self.prev_error = []
        # self.passed_error = []
        # self.error = []
        # self.input = []
        # self.output = []

    def init_params(self):
        return None

    def calculate_delta_output(self):
        # Assuming untuk output layer
        true_val = [1 if i == self.error_calc_output else 0 for i in range(self.class_num)]
        class_errors = []
        for y_true_val, output_val in zip(true_val, self.output):
            
            class_error_term = 0
            if (self.function_name == "sigmoid"):
                # Buat hidden tinggal ambil term: output_val * (1 - output_val) 
                class_error_term = (y_true_val - output_val) * output_val * (1 - output_val) 
            elif (self.function_name == "relu"):
                # Buat hidden tinggal ambil term: relu_derivative_val
                relu_derivative_val = 1 if output_val > 0 else 0
                class_error_term = (y_true_val - output_val) * relu_derivative_val
            else:
                raise Exception("Invalid activation function name")
        
            class_errors.append(class_error_term)
        
        self.passed_error = np.array(class_errors)
    
    # Assumption only works for layers where output dimension of next layer is equal to activation layer's input dimension
    def calculate_error(self):
        # activation function defaults to relu
        # if (self.function_name == "sigmoid"):
        #     v_activation = np.vectorize(self.d_sigmoid)
        # elif (self.function_name == "relu"):
        #     v_activation = np.vectorize(self.d_relu)
        # else:
        #     raise Exception("Invalid activation function name")

        # self.passed_error = v_activation(self.prev_error)

        result = np.zeros(self.output.shape)

        for channel in range(self.output.shape[-1]):
            for i in range(self.output.shape[0]):
                for j in range(self.output.shape[1]):
                    if(self.output[i,j,channel] > 0):
                        result[i,j,channel] = self.prev_error[i,j,channel]
                    else:
                        result[i,j,channel] = 0
        
        self.passed_error = result.copy()
        
        # Asumsi: 
        #     - error dikali dari yang paling dekat output layer dulu
        #     - prev_error = error layer2 sebelumnya; jika di-iterasi mulai output layer 
        # Maka: delta = prev_error * curr_error; 
        
        # Cari dimensi yang sama dari prev_error
        # if (self.prev_error.shape[1] == curr_error.shape[0]): #n_col(prev_error) == n_rows(curr_error), langsung kali
        #     self.error = np.matmul(self.prev_error, curr_error)
        # elif (self.prev_error.shape[1] != curr_error.shape[0]): # Kalo n_col(prev_error) != n_rows(curr_error), transpose
        #     self.error = np.matmul(self.prev_error, curr_error.transpose())
        # else: # if no intersection
        #     raise Exception("Cannot do matrix multiplication. No intersecting dimensions")

        return None

    def d_sigmoid(self, out):
        # Works for single value only
        return out * (1 - out)

    def d_relu(self, out):
        # Works for single value only
        return 1 if out > 0 else 0

    def sigmoid(self, net):
        return 1. / (1. + np.exp(-net))

    def relu(self, net):
        return 0 if (net < 0) else net

    def run(self):
        function_name = self.function_name

        # activation function defaults to relu
        if (function_name == "sigmoid"):
            v_activation = np.vectorize(self.sigmoid)
        elif (function_name == "relu"):
            v_activation = np.vectorize(self.relu)
        else:
            raise Exception("Invalid activation function name")

        self.output = v_activation(self.input)
        return None

    def backprop(self, error):
        result = np.zeros(self.output.shape)

        for channel in range(self.output.shape[-1]):
            for i in range(self.output.shape[0]):
                for j in range(self.output.shape[1]):
                    if(self.output[i,j,channel] > 0):
                        result[i,j,channel] = error[i,j,channel]
                    else:
                        result[i,j,channel] = 0
        
        return result

    def update_weight(self):
        #No weight to be updated in detection stage
        return None
