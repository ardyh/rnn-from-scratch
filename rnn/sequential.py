class Sequential:
    def __init__(self, input_shape, learning_rate=0.5, momentum=0.1):
        self.input = []
        self.layers = [] 
        self.final_output = []
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.momentum = momentum

    def add(self, layer):
        self.layers.append(layer)
        return None

    def DEBUGforwardprop(self, X_instance):
        # Initialize parameters for every layer
        for layer in self.layers:
            layer.init_params(self.input_shape)
        prev_output = [] 
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                layer.input = X_instance
            else:
                layer.input = prev_output
                
            layer.run()
            prev_output = layer.output.copy()
        
        self.final_output = prev_output

    def forwardprop(self, X_instance):
        prev_output = [] 
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                layer.input = X_instance
            else:
                layer.input = prev_output
                
            layer.run()
            prev_output = layer.output.copy()
        
        self.final_output = prev_output

    def backprop(self, y_instance, is_update):
        prev_error = []; 
        error_calc_output = []
        for idx, layer in enumerate(reversed(self.layers)):
            if idx == 0: # If last layer
                layer.error_calc_output = y_instance
                layer.calculate_delta_output()
            else:
                layer.prev_error = prev_error
                layer.calculate_error()
            
            prev_error = layer.passed_error

        if (is_update):
            for layer in self.layers:
                layer.update_weight()

    def train(self, X, y, epochs=50, batch_size=5):
        instance_size = len(X)
        
        # Initialize parameters for every layer
        for layer in self.layers:
            layer.init_params(self.input_shape)

        # iterate every epoch
        for epoch in list(range(epochs)):
            print(f"Epoch {epoch}")
            # iterate every instance
            for instance_idx, instance in enumerate(zip(X, y)):
                X_instance = instance[0]; y_instance = instance[1]
                
                # If last instance or multiple of batch, update params
                is_update = (instance_idx == instance_size) or (instance_idx % batch_size == 0)
                
                self.forwardprop(X_instance)
                self.backprop(y_instance, is_update)
        return None

    def predict(self, X):
        y_pred = []
        for img in X:
            self.forwardprop(img)
            y_pred.append(np.argmax(self.final_output))

        return y_pred
    