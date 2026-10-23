"""Round-trip test: decode(encode(x)) == x for held-out questions and programs.

Run with `python3 tokenizer/test_tokenizer.py` or `pytest tokenizer/`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizers import Tokenizer

from tokenizer import OUT_PATH, SPECIAL_TOKENS, load_train, split_indices


def load_texts():
    ds = load_train()
    _, _, test_idx = split_indices(len(ds))
    rows = ds.select(test_idx)
    return rows["question"] + rows["code"]


def test_roundtrip():
    tok = Tokenizer.from_file(OUT_PATH)
    texts = load_texts()
    encodings = tok.encode_batch(texts, add_special_tokens=False)
    decoded = tok.decode_batch([e.ids for e in encodings], skip_special_tokens=False)
    failures = [(t, d) for t, d in zip(texts, decoded) if t != d]
    assert not failures, f"{len(failures)}/{len(texts)} mismatches, first: {failures[0]!r}"
    print(f"round-trip OK on {len(texts):,} texts")


def test_specials_and_digits():
    tok = Tokenizer.from_file(OUT_PATH)
    assert [tok.token_to_id(t) for t in SPECIAL_TOKENS] == [0, 1, 2]
    # Every digit is its own token.
    tokens = tok.encode("x = 12345\n    return x", add_special_tokens=False).tokens
    assert [t for t in tokens if any(ch.isdigit() for ch in t)] == list("12345")
    # Special tokens decode to nothing when skipped, and survive otherwise.
    ids = [tok.token_to_id("<bos>")] + tok.encode("a  b\n\tc", add_special_tokens=False).ids
    assert tok.decode(ids, skip_special_tokens=True) == "a  b\n\tc"


if __name__ == "__main__":
    test_specials_and_digits()
    test_roundtrip()
