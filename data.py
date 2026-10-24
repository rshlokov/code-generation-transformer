"""Data loading, tokenization and 2-D length bucketing with per-side token caps."""
import glob
import os

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

REPO = "SlavaYus/tinysms_qa_12m"
TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer", "tokenizer.json")
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2

# Lengths include the appended <eos> (encoder) / <bos> (decoder). Chosen from the length
# distribution of held-out train rows: ~7.7% encoder and ~5.9% decoder padding, ~0.6% dropped.
# The coarse edges (64, 128, ...) would pad ~35% / ~27%.
Q_EDGES = (24, 30, 36, 42, 48, 56, 64, 76, 96, 128)
C_EDGES = (30, 36, 42, 48, 54, 60, 68, 76, 86, 96, 108, 122, 138, 160, 192, 256)


def load_tokenizer():
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    assert [tok.token_to_id(t) for t in ("<pad>", "<bos>", "<eos>")] == [PAD_ID, BOS_ID, EOS_ID]
    return tok


CACHE_ARRAYS = ("q_flat", "q_off", "c_flat", "c_off")


def cache_path(out_dir, n_rows, seed):
    """Local directory of the token cache for n_rows sampled rows (None = all)."""
    return os.path.join(out_dir, f"tokens_{n_rows or 'all'}_seed{seed}")


def _ragged(encodings):
    """List of encodings -> (flat uint16 ids, int64 lengths)."""
    lens = np.fromiter((len(e.ids) for e in encodings), dtype=np.int64, count=len(encodings))
    flat = np.fromiter((i for e in encodings for i in e.ids), dtype=np.uint16, count=int(lens.sum()))
    return flat, lens


def _offsets(lens):
    offsets = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    return offsets


def build_token_cache(path, n_rows=None, seed=0, log=print):
    """Tokenize n_rows uniformly sampled train rows (None = all) into a directory of .npy arrays.

    Reads the parquet shards directly and takes the same fraction of random rows from every
    shard. Each shard is turned into flat uint16 arrays right away, so memory stays at the size
    of the token ids even for all 11.8M rows. Only the train split is touched.
    """
    local = snapshot_download(REPO, repo_type="dataset", allow_patterns=["data/train/*.parquet"])
    shards = sorted(glob.glob(os.path.join(local, "data", "train", "*.parquet")))
    sizes = [pq.ParquetFile(s).metadata.num_rows for s in shards]
    total = sum(sizes)
    take_all = n_rows is None or n_rows >= total
    chosen = None if take_all else np.sort(np.random.default_rng(seed).choice(total, size=n_rows, replace=False))
    tok = load_tokenizer()
    parts = {k: [] for k in ("q_flat", "q_len", "c_flat", "c_len")}
    start = 0
    for n, (shard, size) in enumerate(zip(shards, sizes)):
        table = pq.read_table(shard, columns=["question", "code"])
        if not take_all:
            idx = chosen[(chosen >= start) & (chosen < start + size)] - start
            table = table.take(idx)
        start += size
        for side, col in (("q", "question"), ("c", "code")):
            flat, lens = _ragged(tok.encode_batch(table.column(col).to_pylist(), add_special_tokens=False))
            parts[f"{side}_flat"].append(flat)
            parts[f"{side}_len"].append(lens)
        log(f"tokenized shard {n + 1}/{len(shards)}")
    arrays = {"q_flat": np.concatenate(parts["q_flat"]), "q_off": _offsets(np.concatenate(parts["q_len"])),
              "c_flat": np.concatenate(parts["c_flat"]), "c_off": _offsets(np.concatenate(parts["c_len"]))}
    tmp = path + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    for k in CACHE_ARRAYS:
        np.save(os.path.join(tmp, f"{k}.npy"), arrays[k])
    os.replace(tmp, path)  # atomic: a half-written cache is never picked up
    log(f"tokenized {len(arrays['q_off']) - 1:,} of {total:,} train rows -> {path}")


class TokenizedPairs:
    """Ragged question/code token arrays (no special tokens), memory-mapped from the cache
    directory so the DDP processes share one copy in the page cache."""

    def __init__(self, path):
        d = {k: np.load(os.path.join(path, f"{k}.npy"), mmap_mode="r") for k in CACHE_ARRAYS}
        self.q_flat, self.q_off = d["q_flat"], d["q_off"]
        self.c_flat, self.c_off = d["c_flat"], d["c_off"]

    def __len__(self):
        return len(self.q_off) - 1

    def question(self, i):
        return self.q_flat[self.q_off[i]:self.q_off[i + 1]]

    def code(self, i):
        return self.c_flat[self.c_off[i]:self.c_off[i + 1]]

    @property
    def enc_lens(self):  # question + <eos>
        return np.diff(self.q_off) + 1

    @property
    def dec_lens(self):  # <bos> + code  (== code + <eos> for the labels)
        return np.diff(self.c_off) + 1


class BucketBatcher:
    """Batches from single (question bucket, code bucket) cells under per-side token caps.

    The encoder and decoder sides are capped separately: a decoder token costs about twice the
    memory of an encoder token (it also carries the logits over the vocabulary), so one shared
    budget lets decoder-heavy buckets use far more memory than the rest.

    Every rank builds the same global batch list for an epoch (same seed), then takes every
    world_size-th batch, so ranks see disjoint batches and the same number of steps.
    """

    def __init__(self, data: TokenizedPairs, enc_tokens, dec_tokens, q_edges=Q_EDGES, c_edges=C_EDGES,
                 rank=0, world_size=1, seed=0):
        self.data = data
        self.enc_tokens, self.dec_tokens = enc_tokens, dec_tokens
        self.q_edges, self.c_edges = np.array(q_edges), np.array(c_edges)
        self.rank, self.world_size, self.seed = rank, world_size, seed

        enc, dec = data.enc_lens, data.dec_lens
        keep = (enc <= self.q_edges[-1]) & (dec <= self.c_edges[-1])
        self.n_total, self.n_dropped = len(enc), int((~keep).sum())
        idx = np.nonzero(keep)[0]
        qb = np.searchsorted(self.q_edges, enc[idx])  # first edge >= length
        cb = np.searchsorted(self.c_edges, dec[idx])
        self.buckets = {}
        for key in sorted(set(zip(qb.tolist(), cb.tolist()))):
            sel = (qb == key[0]) & (cb == key[1])
            self.buckets[(int(self.q_edges[key[0]]), int(self.c_edges[key[1]]))] = idx[sel]

    def batch_size(self, q_len, c_len):
        """Examples per batch: padded encoder and decoder tokens each stay within their cap."""
        return max(1, min(self.enc_tokens // q_len, self.dec_tokens // c_len))

    def heaviest_shape(self, dec_weight=2.0):
        """(q_len, c_len, batch_size) of the bucket whose full batch needs the most memory,
        counting a decoder token as dec_weight encoder tokens."""
        def cost(shape):
            q_len, c_len = shape
            return self.batch_size(q_len, c_len) * (q_len + dec_weight * c_len)

        q_len, c_len = max(self.buckets, key=cost)
        return q_len, c_len, self.batch_size(q_len, c_len)

    def epoch_batches(self, epoch):
        """All batches of an epoch for this rank: list of (q_len, c_len, row indices)."""
        rng = np.random.default_rng((self.seed, epoch))
        batches = []
        for (q_len, c_len), rows in self.buckets.items():
            rows = rng.permutation(rows)
            bs = self.batch_size(q_len, c_len)
            batches += [(q_len, c_len, rows[i:i + bs]) for i in range(0, len(rows), bs)]
        order = rng.permutation(len(batches))
        batches = [batches[i] for i in order]
        batches = batches[:len(batches) - len(batches) % self.world_size]
        return batches[self.rank::self.world_size]

    def padding_fraction(self):
        """Expected padding share of the encoder and decoder slots over one epoch."""
        enc_real = enc_slots = dec_real = dec_slots = 0
        enc, dec = self.data.enc_lens, self.data.dec_lens
        for (q_len, c_len), rows in self.buckets.items():
            enc_real += enc[rows].sum(); enc_slots += q_len * len(rows)
            dec_real += dec[rows].sum(); dec_slots += c_len * len(rows)
        return 1 - enc_real / enc_slots, 1 - dec_real / dec_slots

    def collate(self, q_len, c_len, rows):
        B = len(rows)
        enc = np.full((B, q_len), PAD_ID, dtype=np.int64)
        dec = np.full((B, c_len), PAD_ID, dtype=np.int64)
        labels = np.full((B, c_len), -100, dtype=np.int64)
        for j, r in enumerate(rows):
            q, c = self.data.question(r), self.data.code(r)
            enc[j, :len(q)] = q
            enc[j, len(q)] = EOS_ID
            dec[j, 0] = BOS_ID
            dec[j, 1:len(c) + 1] = c
            labels[j, :len(c)] = c
            labels[j, len(c)] = EOS_ID
        enc = torch.from_numpy(enc)
        return {
            "encoder_input_ids": enc,
            "encoder_padding_mask": enc == PAD_ID,  # True = padding
            "decoder_input_ids": torch.from_numpy(dec),
            "labels": torch.from_numpy(labels),
        }

    def iterate(self, start_epoch=0, start_batch=0):
        """Endless stream of (epoch, batch_index, batch); resumable from (epoch, batch)."""
        epoch = start_epoch
        while True:
            for b, (q_len, c_len, rows) in enumerate(self.epoch_batches(epoch)):
                if epoch == start_epoch and b < start_batch:
                    continue
                yield epoch, b, self.collate(q_len, c_len, rows)
            epoch += 1
