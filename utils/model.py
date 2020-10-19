### model code
import numpy as np
from .activations import get_activation_by_name
from .general_functions import weights_init_random

class Layer():
    def __init__(self, input_size, output_size, activation, is_first = False, is_last = False, init = weights_init_random):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = get_activation_by_name(activation) ### May be None if not implemented
        self.__activation_name = activation
        self.is_first = is_first
        self.is_last = is_last
        self.last_input = None
        self.weighted_sum = None
        self.weighted_sum_activated = None
        self.layer_type = None

        self.W = init(input_size*output_size, 1)
        self.b = init(output_size, 1)[0]

    def update_weight(self, w_id, lr, dW):
        self.W[w_id] = self.W[w_id] - lr * dW
    
    def get_params(self):
        return self.W, self.b

    def __repr__(self):
        return 'Layer of type '+self.layer_type
    
    def __str__(self):
        return 'Layer of type {} with shape: [{},{}], activation: {}, first: {}, last: {}'.format(self.layer_type, 
        self.input_size, self.output_size, self.__activation_name, self.is_first, self.is_last)

class Dense(Layer):
    def __init__(self, input_size, output_size, activation, is_first = False, is_last = False, init = weights_init_random):
        super().__init__(input_size, output_size, activation, is_first, is_last, init)
        self.layer_type = 'Dense'
        
    def compute_weighted_sum(self, Z):
        self.last_input = Z.copy()
        if self.is_first:
            self.weighted_sum = Z
            return Z
        #if self.is_last:
        #    return ### Return cost func output # TODO
        # If this layer is not the first one, we gotta calculate weighted sum and return it
        out = [] ### TODO: replace it with a numpy array
        for i, idz in enumerate(Z): ### Compute the weighted sum
            for idn in range(self.output_size):
                w_id = i*self.output_size + idn
                if i == 0: # if we haven't went throw all neurons yet
                    out.append(idz * self.W[w_id])
                else:
                    out[idn]+= idz * self.W[w_id]
        out = np.array(out)
        self.weighted_sum = out #np.sum(out)
        print('compute_weighted_sum:',self.weighted_sum)
        return self.weighted_sum
    
    def compute_activation(self):
        #print(self.weighted_sum)
        if self.weighted_sum is not None:
            self.weighted_sum_activated = self.activation[0](self.weighted_sum)
            return self.weighted_sum_activated
        else:
            raise Exception('The weighted sum has not been computed yet!')

class Numpy_nn():
    def __init__(self):
        self.layers = {}
    
    def forward(self, Z):
        for layer in self.layers.values():
            #print('---')
            Z = layer.compute_weighted_sum(Z)
            #print('---+')
            Z = layer.compute_activation()
            #print('---++')
        return Z

    def backward(self):
        return

    def __str__(self):
        model_descr = 'Model parameters: \n'
        for layer_name, layer in self.layers.items():
            model_descr+='\t'+str(layer)+', \n'
        model_descr = model_descr[0:-3]
        return model_descr

    def get_state_dict(self):
        return [layer.get_params() for layer in self.layers.values()]

    def save_model(self):
        pass
   
def get_layer_by_name(layer_name):
    if layer_name == 'dense':
        return Dense
    else:
        raise Exception('There are no layer type named ',layer_name,'.', sep='')




