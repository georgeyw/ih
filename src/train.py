import json
import os


def _read_train_config(config_name) -> dict:
    if not config_name.endswith('.json'):
        config_name += '.json'
    path = os.path.join(os.path.dirname(__file__), 'train_configs', config_name)
    with open(path, 'r') as f:
        config = json.load(f)
        return config
