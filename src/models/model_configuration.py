import os
import yaml
import json

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
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

    def __getattr__(self, attr):
        return self.config.get(attr, None)

    def __setattr__(self, attr, value):
        if attr == "config":
            super().__setattr__(attr, value)
        else:
            self.config[attr] = value

    def to_json_string(self):
        return json.dumps(self.config, sort_keys=True)