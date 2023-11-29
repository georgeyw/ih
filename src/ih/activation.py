import torch
from transformer_lens import HookedTransformer

from ih.dgp import DGP


def run_sample_with_cache(dgp: DGP, model: HookedTransformer, layer_idx: str):
    sample, i_seq = dgp._generate_sample(return_i_seq=True)
    layer = f'blocks.{layer_idx}.attn.hook_pattern'
    _, cache = model.run_with_cache(sample)
    return cache[layer][0].permute(1, 2, 0), _to_str_tokens(sample), i_seq


def _to_str_tokens(sample: torch.tensor):
    str_sample = []
    for token in sample:
        str_sample.append(str(token.item()))
    return str_sample
