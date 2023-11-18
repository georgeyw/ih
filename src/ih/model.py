import json
import os
import torch.nn as nn

from transformer_lens import HookedTransformer, HookedTransformerConfig


def build_model(config_name: str = None, **kwargs) -> nn.Module:
    if config_name is None:
        config_name = 'default.json'
    config = _read_model_config(config_name)
    config.update(kwargs)
    _validate_config(config)
    model = HookedTransformer(HookedTransformerConfig(**config))
    return model

def _validate_config(config: dict):
    if not 'n_layers' in config:
        raise ValueError('n_layers must be specified in either config or kwargs')

def _read_model_config(config_name) -> dict:
    if not config_name.endswith('.json'):
        config_name += '.json'
    path = os.path.join(os.path.dirname(__file__), 'model_configs', config_name)
    with open(path, 'r') as f:
        config = json.load(f)
        return config
    