from utils.model import *
from utils.general_functions import *
from utils.activations import *
from utils.cost_functions import *
from utils.config_reader import Config_reader

config_path = 'model_config.json'

config_reader = Config_reader(config_path)
config = config_reader.get_config()

model = Numpy_nn()
for i, (layer_name, layer_params) in enumerate(config['layers'].items()):
    layer = get_layer_by_name(layer_params[-1])(layer_params[0],
                                layer_params[1], activation=layer_params[-2],
                                is_first= i == 0, is_last= i == len(config['layers'])-1)
    model.layers[layer_name] = layer
    print(i == 0, i == len(config['layers'])-1)

def forward_pass():
    return ''

def backward_pass():
    return ''

if __name__ == '__main__':
    print(model)