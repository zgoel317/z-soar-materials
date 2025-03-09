from typing import Any
import random

import torch
from jaxtyping import Int
from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import pytest

from delphi.latents import (
    LatentRecord, Latent, ActivatingExample, LatentDataset, LatentCache, constructor
)
from delphi.latents.latents import ActivationData
from delphi.latents.cache import get_nonzeros_batch
from delphi.config import ConstructorConfig, SamplerConfig


def test_save_load_cache(cache_setup: dict[str, Any], tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast):
    sampler_cfg = SamplerConfig(
        n_examples_train=3,
        n_examples_test=3,
        n_quantiles=3,
        train_type="quantiles",
        test_type="quantiles"
    )
    dataset = LatentDataset(
        cache_setup["temp_dir"],
        sampler_cfg,
        ConstructorConfig(),
        tokenizer
    )
    tokens: Int[Tensor, "examples ctx_len"] = dataset.load_tokens()  # type: ignore
    assert (tokens == cache_setup["tokens"][:len(tokens)]).all()
    for record in dataset:
        assert len(record.train) <= sampler_cfg.n_examples_train
        assert len(record.test) <= sampler_cfg.n_examples_test


@pytest.fixture(scope="module")
def seed():
    random.seed(0)
    torch.manual_seed(0)

@pytest.mark.parametrize("n_samples", [5, 10, 100, 1000])
@pytest.mark.parametrize("n_quantiles", [2, 5, 10, 23])
def test_simple_cache(n_samples: int, n_quantiles: int,
                      ctx_len: int = 32,
                      seed: None = None, *, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast):
    tokens = torch.randint(0, 100, (n_samples, ctx_len,))
    all_activation_data = []
    all_activations = []
    for _ in range(2):
        activations = torch.rand(n_samples, ctx_len) * (torch.rand(n_samples)[..., None] ** 2)
        all_activations.append(activations)
        mask = activations > 0.1
        all_activation_data.append(ActivationData(torch.nonzero(mask), activations[mask]))
    activation_data, other_activation_data = all_activation_data
    activations, other_activations = all_activations
    record = LatentRecord(
        latent=Latent("test", 0),
        examples=[]
    )
    constructor(
        record,
        activation_data,
        constructor_cfg=ConstructorConfig(
            example_ctx_len=ctx_len,
            min_examples=1,
            max_examples=100,
            n_non_activating=50,
            non_activating_source="neighbours",
        ),
        tokens=tokens,
        tokenizer=tokenizer,
        all_data={1: other_activation_data}
    )
    for i, j in zip(record.examples[:-1], record.examples[1:]):
        assert i.max_activation >= j.max_activation
    for i in record.examples:
        index = (tokens == i.tokens).all(dim=-1).float().argmax()
        assert (tokens[index] == i.tokens).all()
        assert activations[index].max() == i.max_activation