import numpy as np

class Sequential:
    def __init__(self):
        self.input = []
        self.layers = [] 
        self.final_output = []

    # Append layer into the model
    def add(self, layer):
        self.layers.append(layer)
        return None
    
    # Initialize parameters for every layer
    def init_layers_param(self):
        for idx, layer in enumerate(self.layers):
            # If the attribute input_shape in the first layer is None, something is wrong
            if (idx == 0 and (not layer.input_shape)):
                raise Exception("First layer must have input_shape property")
            layer.init_params()

    # Reset every layer's hidden unit value (output)
    def reset_layers_output(self):
        for layer in (self.layers):
            layer.reset_output()

    # Execute forward propagation for an instance
    def forwardprop(self, X_instance):
        prev_output = None 
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                layer.input = X_instance
            else:
                layer.input = prev_output
                
            layer.run()
            prev_output = layer.output.copy()
        
        self.final_output = prev_output

    # Train the model. process all instances
    def train(self, X, timestep=3):
        instance_size = len(X)
        self.init_layers_param()
        
        timestep_counter = 0
        # iterate every timestep and instance
        for instance_idx, instance in enumerate(X):
            # if timestep counter resets, re-init layers param
            if timestep_counter == 0:
                self.reset_layers_output()
            
            self.forwardprop(instance)

            timestep_counter += 1
            # if timestep counter exceeds number of timestep, reset the counter
            if timestep_counter == timestep: timestep_counter = 0

        return None
