import os
import yaml

class ModelConfiguration:
    """Provides configuration related to model architecture and training hyperparameters.

    This class holds configuration used to determine the model's architecture
    such as the number of hidden layers, dropout usage, etc.  It also holds hyperperameters
    used for training, such as batch size, learning rate etc.

    These parameters are loaded from a YAML configuration file at
    initialization. The configuration file is assumed to be at
    a relative path from this source file: '../../model-config.yaml'
    """

    def __init__(self):
        config_file = os.path.join(os.path.dirname(__file__), '../../model-config.yaml')
        self.config = self.load_config(config_file)

    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def __getattr__(self, attr):
        return self.config.get(attr, None)

