import json
import os
import warnings
import torch.nn as nn

from transformer_lens import HookedTransformer, HookedTransformerConfig


def build_model(config_name: str = None, **kwargs) -> nn.Module:
    if config_name is None:
        config_name = 'default.json'
    config = _read_model_config(config_name)

    original_config = config.copy()
    config.update(kwargs)
    if config != original_config:
        # messes with wandb logging and makes it hard to record experiments
        warnings.warn("WARNING: overriding config with kwargs. This should only be done for "
                      "debugging, not for recorded experiments.")
        warnings.warn(f"   - original config: {original_config}")
        warnings.warn(f"   - kwargs: {kwargs}")
        warnings.warn(f"   - new config: {config}")

    model = HookedTransformer(HookedTransformerConfig(**config))
    return model


def _read_model_config(config_name) -> dict:
    if not config_name.endswith('.json'):
        config_name += '.json'
    path = os.path.join(os.path.dirname(__file__), 'model_configs', config_name)
    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        return config
