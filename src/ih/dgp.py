from typing import List

import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from ih.utils import read_dgp_config


# generates ~1.5M tokens per second
# do I want this to be strings or ints? I think ints
class DGP:
    def __init__(self,
                 ctx_length: int = 32,
                 num_tokens: int = 10,
                 bigrams: List[List[str]] = None,
                 trigrams: List[List[str]] = None,
                 induction_length: int = 3,
                 bigram_freq: float = 0.1,
                 trigram_freq: float = 0.1,
                 induction_freq: float = 0.1,
                 seed: int = None):
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
        self._validate_params()

    def generate_dataset(self, num_samples):
        samples = self.generate_samples(num_samples)
        dataset = TensorDataset(samples)
        return dataset

    def generate_samples(self, num_samples):
        if self.seed is None:
            samples = [self._generate_sample() for _ in range(num_samples)]
        else:
            samples = [self._generate_sample(
                seed=self.seed + i) for i in range(num_samples)]
        return torch.stack(samples)

    def _generate_sample(self, seed=None):
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
        return torch.tensor(sequence)

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


def build_dgp_for_model(model: nn.Module, dgp_config_name: str = None) -> DGP:
    if dgp_config_name is None:
        dgp_config_name = 'default.json'
    dgp_config = read_dgp_config(dgp_config_name)
    dgp_config['ctx_length'] = model.config['n_ctx']
    dgp_config['num_tokens'] = model.config['d_vocab']
    dgp_config['seed'] = model.config['seed']
    dgp = DGP(**dgp_config)
    return dgp
