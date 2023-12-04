from typing import Union
import warnings
import torch.nn as nn

from transformer_lens import HookedTransformer, HookedTransformerConfig
from ih.utils import read_config


def build_model(config: Union[str, dict] = None, device = 'cpu', **kwargs) -> nn.Module:
    if config is None:
        config = 'default-L1.json'
    if isinstance(config, str):
        config = read_config(config, 'model')
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
