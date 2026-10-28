"""Fixed-context workload construction and latency reduction."""

import math

import pytest

from ignition.bench.runner import BenchmarkCase, prompt_tokens, summarize_case


def test_case_estimate_keeps_raw_samples_and_uses_all_seeds():
    case = BenchmarkCase(2, 8, 16, 512)
    samples = [dict(seed=0, times_ms=[1., 3.]), dict(seed=1, times_ms=[2., 10.])]
    result = summarize_case(case, samples)
    assert result["median_ms"] == 2.5
    assert result["p25_ms"] == 1.75
    assert result["p75_ms"] == 4.75
    assert result["relative_iqr"] == 1.2
    assert result["samples"] is samples
    assert (result["batch"], result["packed_tokens"], result["context_tokens"]) == (2, 16, 512)


@pytest.mark.parametrize("times", [[], [0.], [-1.], [math.inf], [math.nan]])
def test_missing_or_invalid_measurements_are_not_latency_estimates(times):
    with pytest.raises(ValueError, match="finite and positive"):
        summarize_case(BenchmarkCase(1, 8, 8, 512), [dict(seed=0, times_ms=times)])


@pytest.mark.parametrize("arguments", [(3, 8, 24, 512), (1, 4, 4, 512), (2, 8, 8, 512), (1, 8, 8, 0)])
def test_case_rejects_shapes_outside_complete_round_matrix(arguments):
    with pytest.raises(ValueError):
        BenchmarkCase(*arguments)


def test_prompts_are_exact_tokenized_prefixes_and_seeded():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.train_from_iterator([
        "Read observations about database recovery, orbital mechanics, bounded queues and seasonal change. "
        "Record assumptions, measurements, uncertainty and independent conclusions."
    ], trainers.BpeTrainer(vocab_size=300, special_tokens=["[UNK]"]))
    first = prompt_tokens(tokenizer, 512, seed=0, lane=0)
    assert len(first) == 512
    assert first == prompt_tokens(tokenizer, 512, seed=0, lane=0)
    assert first != prompt_tokens(tokenizer, 512, seed=1, lane=0)
    assert first != prompt_tokens(tokenizer, 512, seed=0, lane=1)
    assert first == prompt_tokens(tokenizer, 2048, seed=0, lane=0)[:512]
