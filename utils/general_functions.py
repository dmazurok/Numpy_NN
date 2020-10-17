import numpy as np

def weights_init(model_params): # model_params: [{input_dim:N, output_dim:N}...]
    params_values = {}

    for idx, layer in enumerate(model_params):
        layer_idx = idx + 1
        layer_output_size = layer['output_dim']
    
        params_values['W_' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1
        params_values['b_'+str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1
    
    return params_values

def weights_init_random(size, dim):
    return np.random.randn(size, dim) * 0.1