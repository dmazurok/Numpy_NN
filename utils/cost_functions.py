import numpy as np 

def mse_loss(target:np.array, y:np.array):
    return np.sum((target-y)**2)*0.5
    
def mse_loss_backward(target:float, y:float):
    return -target+y

def cross_entropy_loss(target:np.array, y:np.array): ### to be done
    return 