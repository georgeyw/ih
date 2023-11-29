import warnings
import torch.nn as nn

from transformer_lens import HookedTransformer, HookedTransformerConfig
from ih.utils import read_model_config


def build_model(config_name: str = None, device = 'cpu', **kwargs) -> nn.Module:
    if config_name is None:
        config_name = 'default-L1.json'
    config = read_model_config(config_name)
    config['device'] = device

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
