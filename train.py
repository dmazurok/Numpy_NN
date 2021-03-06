from utils.model import *
from utils.general_functions import *
from utils.activations import *
from utils.cost_functions import *
from utils.config_reader import Config_reader

config_path = 'model_config.json'

epochs = 1

lr = 0.1

config_reader = Config_reader(config_path)
config = config_reader.get_config()

model = Numpy_nn()
for i, (layer_name, layer_params) in enumerate(config['layers'].items()):
    layer = get_layer_by_name(layer_params[-1])(layer_params[0],
                                layer_params[1], activation=layer_params[-2],
                                is_first= i == 0, is_last= i == len(config['layers'])-1)
    model.layers[layer_name] = layer

if __name__ == '__main__':
    for epoch in range(epochs):
        y_ = model.forward(np.random.randn(2,1), np.random.randn(2,1))
        print('OUT:',y_[0])
        y = np.array([1.,0.])
        print('loss is',mse_loss(y_[0], y))
        print('Predicted:',model.backward(y, lr))
