"""Model tests on CPU in fp32. Run with `python3 tests/test_model.py` or `pytest tests/`."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import BucketBatcher, EOS_ID, PAD_ID  # noqa: E402
from models.model import ModelConfig, Seq2SeqTransformer, greedy_generate  # noqa: E402


def small_model(vocab_size=64, seed=0, std=0.3):
    """Larger-than-default init so a random model's greedy outputs differ between questions."""
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=vocab_size, d_model=64, n_heads=2, max_len=256)
    model = Seq2SeqTransformer(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim >= 2:
                p.normal_(0, std)
    return model


def random_seq(rng, n, vocab):
    return torch.from_numpy(rng.integers(3, vocab, size=n))


def pad_batch(seqs):
    out = torch.full((len(seqs), max(len(s) for s in seqs)), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out


@torch.no_grad()
def test_padding_invariance():
    """An example's logits are the same batched with padding as run alone."""
    model = small_model(std=0.02)  # standard init: logits of order 1, so fp32 noise stays ~1e-6
    rng = np.random.default_rng(0)
    enc_lens, dec_lens = [5, 17, 11], [9, 3, 20]
    encs = [random_seq(rng, n, 64) for n in enc_lens]
    decs = [random_seq(rng, n, 64) for n in dec_lens]
    enc, dec = pad_batch(encs), pad_batch(decs)
    batched = model(enc, enc == PAD_ID, dec)
    for i in range(len(encs)):
        e, d = encs[i][None], decs[i][None]
        alone = model(e, e == PAD_ID, d)[0]
        torch.testing.assert_close(batched[i, :dec_lens[i]], alone, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_kv_cache_matches_uncached():
    """Greedy generation with the KV cache gives exactly the same tokens as without it."""
    # Small vocab so that <eos> comes up and sequences finish at different steps.
    model = small_model(vocab_size=8, seed=0)
    rng = np.random.default_rng(1)
    encs = [random_seq(rng, n, 8) for n in rng.integers(3, 30, size=16)]
    enc = pad_batch(encs)
    mask = enc == PAD_ID
    ref = greedy_generate(model, enc, mask, max_new_tokens=60, use_cache=False)
    for compact in (True, False):
        got = greedy_generate(model, enc, mask, max_new_tokens=60, use_cache=True, compact=compact)
        assert got == ref
    lengths = {len(r) for r in ref}
    assert len(lengths) > 1, "test should cover sequences finishing at different steps"
    # The batch result also matches generating each question alone.
    for i, e in enumerate(encs):
        e = e[None]
        assert greedy_generate(model, e, e == PAD_ID, max_new_tokens=60)[0] == ref[i]


@torch.no_grad()
def test_deadline_returns_partial():
    model = small_model()
    enc = torch.randint(3, 64, (4, 10))
    out = greedy_generate(model, enc, enc == PAD_ID, max_new_tokens=50, deadline=0.0)
    assert out == [[], [], [], []]


class _FakeData:
    def __init__(self, q_lens, c_lens, rng):
        self.qs = [rng.integers(3, 50, size=n) for n in q_lens]
        self.cs = [rng.integers(3, 50, size=n) for n in c_lens]
        self.enc_lens, self.dec_lens = np.array(q_lens) + 1, np.array(c_lens) + 1

    def question(self, i):
        return self.qs[i]

    def code(self, i):
        return self.cs[i]


def test_bucket_batches():
    rng = np.random.default_rng(0)
    q_lens, c_lens = rng.integers(5, 140, size=3000), rng.integers(5, 270, size=3000)
    data = _FakeData(q_lens, c_lens, rng)
    budget = 4096
    seen = []
    for rank in range(2):
        b = BucketBatcher(data, budget, rank=rank, world_size=2)
        assert b.n_dropped == int(((q_lens + 1 > 128) | (c_lens + 1 > 256)).sum())
        for q_len, c_len, rows in b.epoch_batches(0):
            assert len(rows) * (q_len + c_len) <= budget
            batch = b.collate(q_len, c_len, rows)
            assert batch["encoder_input_ids"].shape == (len(rows), q_len)
            assert batch["decoder_input_ids"].shape == batch["labels"].shape == (len(rows), c_len)
            for j, r in enumerate(rows):
                nq, nc = len(data.qs[r]), len(data.cs[r])
                assert nq + 1 <= q_len and nc + 1 <= c_len  # single bucket on both sides
                assert batch["encoder_input_ids"][j, nq] == EOS_ID
                assert batch["encoder_padding_mask"][j].sum() == q_len - nq - 1
                assert batch["labels"][j, nc] == EOS_ID and (batch["labels"][j, nc + 1:] == -100).all()
                assert (batch["decoder_input_ids"][j, 1:nc + 1] == batch["labels"][j, :nc]).all()
            seen += rows.tolist()
    assert len(seen) == len(set(seen))  # ranks get disjoint examples
    # Different epochs are shuffled differently.
    b = BucketBatcher(data, budget)
    assert [r.tolist() for *_, r in b.epoch_batches(0)] != [r.tolist() for *_, r in b.epoch_batches(1)]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
