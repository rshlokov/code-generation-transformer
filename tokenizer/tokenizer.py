"""Train a shared byte-level BPE tokenizer for questions and code.

Usage: python3 tokenizer/tokenizer.py
Writes tokenizer/tokenizer.json and prints token-length statistics.
"""
import os

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

REPO = "SlavaYus/tinysms_qa_12m"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer.json")

VOCAB_SIZE = 16_384
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]  # ids 0, 1, 2
SEED = 0
N_TRAIN = 1_500_000  # rows used to fit the BPE merges
N_STATS = 200_000    # held-out train rows for length statistics
N_ROUNDTRIP = 5_000  # held-out train rows for the round-trip test


def load_train():
    # Only the train split is ever loaded; the test split is off limits.
    return load_dataset(REPO, data_files="data/train/*.parquet", split="train")


def split_indices(n_rows):
    """Disjoint random row indices for training, statistics and the round-trip test."""
    perm = np.random.default_rng(SEED).permutation(n_rows)
    bounds = np.cumsum([0, N_TRAIN, N_STATS, N_ROUNDTRIP])
    return [np.sort(perm[lo:hi]) for lo, hi in zip(bounds[:-1], bounds[1:])]


def build_tokenizer():
    tok = Tokenizer(models.BPE())
    # No normalizer: text is kept byte-for-byte (case, whitespace, newlines, indentation).
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        # add_prefix_space=False so decode(encode(x)) == x without a leading space.
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def iter_texts(ds, batch_size=10_000):
    for start in range(0, len(ds), batch_size):
        batch = ds[start:start + batch_size]
        yield batch["question"]
        yield batch["code"]


def train(ds):
    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(iter_texts(ds), trainer=trainer, length=2 * len(ds))
    return tok


def lengths(tok, texts):
    return np.array([len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False)])


def report(tok, ds):
    q = lengths(tok, ds["question"])
    c = lengths(tok, ds["code"])
    print(f"Length stats over {len(ds):,} held-out train rows (tokens, no special tokens):")
    for name, x in [("question", q), ("code", c)]:
        print(f"  {name:8s}  mean {x.mean():7.1f}  median {np.median(x):6.0f}  "
              f"p99 {np.percentile(x, 99):6.0f}  max {x.max():6d}")
    print(f"  avg total tokens per example (question + code): {(q + c).mean():.1f}")


def main():
    ds = load_train()
    train_idx, stats_idx, _ = split_indices(len(ds))
    tok = train(ds.select(train_idx))
    assert tok.get_vocab_size() == VOCAB_SIZE
    assert [tok.token_to_id(t) for t in SPECIAL_TOKENS] == [0, 1, 2]
    tok.save(OUT_PATH)
    print(f"Saved {OUT_PATH} (vocab {tok.get_vocab_size()})")
    report(tok, ds.select(stats_idx))


if __name__ == "__main__":
    main()

