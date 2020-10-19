### model code
import numpy as np
from .activations import get_activation_by_name
from .general_functions import weights_init_random
from .cost_functions import *

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
        print('----- UPDATE WEIGHT',w_id,'NOW IT IS',self.W[w_id], dW, lr)
        self.W[w_id] = self.W[w_id] - lr * dW
        print('----- AND NOW IT IS',self.W[w_id])
    
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
        self.target = None
        self.Z = None
    
    def forward(self, Z, target):
        self.target = target
        for layer in self.layers.values():
            #print('---')
            Z = layer.compute_weighted_sum(Z)
            #print('---+')
            Z = layer.compute_activation()
            self.Z = Z.copy()
            #print('---++')
        #print('loss is', mse_loss(target, Z))
        return Z

    def backward(self, predicted, lr):
        out = []
        print('Backward pass. Z=',self.Z)
        for layer_id, layer_name in enumerate(reversed(list(self.layers.keys()))):
            layer = self.layers[layer_name]
            print('\n\t###Layer',layer_name,'W.shape=[',layer.W.shape,'], layer.weighted_sum_activated.shape=[',layer.weighted_sum_activated.shape,']\n')
            out_layer = []
            for i, z in enumerate(layer.weighted_sum_activated):
                z = z.item()
                print('i=',i,'z=',z)
                for wi in range(layer.last_input.shape[0]):
                    print('\t --- X',wi,'\n')
                    # for each weight we need to calculate dC/dy
                    dC_dy = mse_loss_backward(self.target[i], z) if layer.is_last else 1.
                    print('dC_dy:',dC_dy)
                    d_act_Y_dw = layer.activation[1](layer.weighted_sum[i])
                    print('d_act_Y_dw:', d_act_Y_dw)
                    dW_dy = layer.last_input[wi]
                    print('dW_dy:',dW_dy)
                    dC_dw = dC_dy*d_act_Y_dw*dW_dy
                    out_layer.append(dC_dw)
                    layer.update_weight(i*layer.last_input.shape[0]+wi, lr, dC_dw)
            out.append(out_layer)
        return np.array(out)

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




