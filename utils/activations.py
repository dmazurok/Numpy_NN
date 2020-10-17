import numpy as np
def relu(Z):
    return np.maximum(0, Z) 
def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

### Partial derivatives are below

def relu_backward(Z):
    return 0 if Z < 0 else 1
def sigmoid_backward(Z):
    return sigmoid(Z)*(1-sigmoid(Z))

def get_activation_by_name(activation_func_name):
    if activation_func_name == 'relu':
        return relu, relu_backward
    if activation_func_name == 'sigmoid':
        return sigmoid, sigmoid_backward
    raise Exception('There are no activation function named ',activation_func_name,'.', sep='')