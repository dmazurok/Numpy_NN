import json

class config_reader():
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = None
    def get_config(self):
        if not self.config:
            self.config = json.load(open(self.config_file, 'r'))
        return self.config
    def get_model_param_n_per_layer(self): # model_params: [{input_dim:N, output_dim:N}...]
        config = self.get_config()
        return [config['layers'][i][0:2] for i in config['layers']]