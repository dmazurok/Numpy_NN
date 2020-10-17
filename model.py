### model code
import numpy as np
from activations import get_activation_by_name
class Dense():
    def __init__(self, input_size, output_size, activation, init = True):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = get_activation_by_name() ### May be None if not implemented

        self.W = np.array()
