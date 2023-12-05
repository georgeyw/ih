from typing import List, Union

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ih.utils import read_config


# generates ~1.5M tokens per second
# do I want this to be strings or ints? I think ints
class DGP(Dataset):
    def __init__(self,
                 num_samples: int,
                 ctx_length: int = 32,
                 num_tokens: int = 10,
                 bigrams: List[List[int]] = None,
                 trigrams: List[List[int]] = None,
                 induction_length: int = 3,
                 bigram_freq: float = 0.1,
                 trigram_freq: float = 0.1,
                 induction_freq: float = 0.1,
                 seed: int = None,
                 device: str = 'cpu'):
        self.num_samples = num_samples
        self.ctx_length = ctx_length
        self.num_tokens = num_tokens
        self.alphabet = list(range(num_tokens))

        self.bigrams = bigrams or [[self.alphabet[0], self.alphabet[1]],
                                   [self.alphabet[3], self.alphabet[4]]]
        self.trigrams = trigrams or [
            [self.alphabet[2], self.alphabet[3], self.alphabet[5]]]
        self.induction_length = induction_length

        self.bigram_freq = bigram_freq
        self.trigram_freq = trigram_freq
        self.induction_freq = induction_freq

        self.seed = seed
        self.device = device
        self._validate_params()

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if self.seed:
            return self._generate_sample(seed=self.seed)
        return self._generate_sample()

    def _generate_sample(self, return_i_seq: bool = False, seed: int = None):
        random.seed(seed)
        induction_seq = [random.choice(self.alphabet)
                         for _ in range(self.induction_length)]
        sequence = []
        i = 0
        while i < self.ctx_length:
            rand_val = random.random()
            if rand_val < self.bigram_freq:
                new_tokens = random.choice(self.bigrams)
                i += 2
            elif rand_val < self.bigram_freq + self.trigram_freq:
                new_tokens = random.choice(self.trigrams)
                i += 3
            elif rand_val < self.bigram_freq + self.trigram_freq + self.induction_freq:
                new_tokens = induction_seq
                i += self.induction_length
            else:
                new_tokens = [random.choice(self.alphabet)]
                i += 1
            sequence += new_tokens
        sequence = sequence[:self.ctx_length]

        if return_i_seq:
            return torch.tensor(sequence).to(self.device), torch.tensor(induction_seq).to(self.device)
        return torch.tensor(sequence).to(self.device)

    def _validate_params(self):
        assert self.ctx_length > 0
        assert self.induction_length > 0

        assert self.bigram_freq + self.trigram_freq + self.induction_freq <= 1
        assert self.bigram_freq >= 0
        assert self.trigram_freq >= 0
        assert self.induction_freq >= 0

        if self.bigram_freq > 0:
            assert len(self.bigrams) > 0
            for bigram in self.bigrams:
                assert len(bigram) == 2
                assert bigram[0] in self.alphabet
                assert bigram[1] in self.alphabet

        if self.trigram_freq > 0:
            assert len(self.trigrams) > 0
            for trigram in self.trigrams:
                assert len(trigram) == 3
                assert trigram[0] in self.alphabet
                assert trigram[1] in self.alphabet
                assert trigram[2] in self.alphabet


def build_dgp_for_model(model: nn.Module, dgp_config: Union[str, dict], num_samples: int, seed: int = None) -> DGP:
    if isinstance(dgp_config, str):
        dgp_config = read_config(dgp_config, 'dgp')
    dgp_config['num_samples'] = num_samples
    dgp_config['ctx_length'] = model.cfg.n_ctx
    dgp_config['num_tokens'] = model.cfg.d_vocab
    dgp_config['seed'] = model.cfg.seed
    dgp = DGP(**dgp_config)
    return dgp
