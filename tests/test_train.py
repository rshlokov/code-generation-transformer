"""Tests for checkpoint storage, resume checks and the token cache, on CPU without network.
Run with `python3 tests/test_train.py` or `pytest tests/`."""
import os
import sys
import tempfile
import types
from argparse import Namespace
from contextlib import contextmanager

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data  # noqa: E402
import wandb_io  # noqa: E402
from train import DATA_ARGS, SCHEDULE_ARGS, check_resume  # noqa: E402


# ---------------------------------------------------------------------------- fake wandb

class CommError(Exception):
    pass


class FakeArtifact:
    def __init__(self, name, step, files=None):
        self.name, self.metadata, self.files = name, {"step": step}, files or {}
        self.version = name.rsplit(":v", 1)[-1] if ":v" in name else "v0"
        self.deleted = False

    def download(self, root, skip_cache=None):
        os.makedirs(root, exist_ok=True)
        for fname, payload in self.files.items():
            torch.save(payload, os.path.join(root, fname))
        return root

    def delete(self, delete_aliases=False):
        self.deleted = True


@contextmanager
def fake_wandb(latest=None, error=None, versions=()):
    """wandb whose Api().artifact() returns `latest`, raises `error`, or reports "not found"."""
    class Api:
        def artifact(self, name, type=None):
            if error is not None:
                raise error
            if latest is None:
                raise CommError(f"artifact membership '{name}' not found in 'entity/project'")
            return latest

        def artifacts(self, type_name, name):
            return list(versions)

    logged = []

    class Artifact:
        def __init__(self, name, type, metadata=None):
            self.name, self.metadata = name, metadata

        def add_file(self, path, name=None):
            pass

    module = types.SimpleNamespace(Api=Api, Artifact=Artifact, errors=types.SimpleNamespace(CommError=CommError))
    saved = sys.modules.get("wandb")
    sys.modules["wandb"] = module
    try:
        yield logged
    finally:
        if saved is None:
            del sys.modules["wandb"]
        else:
            sys.modules["wandb"] = saved


class FakeRun:
    entity, project = "entity", "project"

    def __init__(self):
        self.logged = []

    def log_artifact(self, art, aliases=None):
        self.logged.append(art)
        return types.SimpleNamespace(wait=lambda: None)


def quiet(msg):
    pass


# ---------------------------------------------------------------------------- checkpoints

def test_fetch_not_found_keeps_local():
    with tempfile.TemporaryDirectory() as d, fake_wandb(latest=None):
        path = os.path.join(d, "ckpt.pt")
        wandb_io.fetch_checkpoint("project", "exp", path, quiet)
        assert not os.path.exists(path)


def test_fetch_network_error_raises():
    """A server or network error must stop the run, never read as "start from scratch"."""
    with tempfile.TemporaryDirectory() as d, fake_wandb(error=CommError("502 Bad Gateway")):
        try:
            wandb_io.fetch_checkpoint("project", "exp", os.path.join(d, "ckpt.pt"), quiet)
        except CommError:
            return
        raise AssertionError("network error was swallowed")


def test_fetch_prefers_newer():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        torch.save({"step": 50}, path)
        remote_old = FakeArtifact("ckpt-exp:v0", 40, {"ckpt.pt": {"step": 40}})
        with fake_wandb(latest=remote_old):
            wandb_io.fetch_checkpoint("project", "exp", path, quiet)
        assert torch.load(path)["step"] == 50  # local is newer: kept
        remote_new = FakeArtifact("ckpt-exp:v1", 90, {"ckpt.pt": {"step": 90}})
        with fake_wandb(latest=remote_new):
            wandb_io.fetch_checkpoint("project", "exp", path, quiet)
        assert torch.load(path)["step"] == 90  # wandb is newer: downloaded


def test_upload_refuses_to_replace_later_step():
    with tempfile.TemporaryDirectory() as d, fake_wandb(latest=FakeArtifact("ckpt-exp:v3", 100)):
        path = os.path.join(d, "ckpt.pt")
        torch.save({"step": 10}, path)
        run = FakeRun()
        try:
            wandb_io.upload_checkpoint(run, path, "exp", 10, keep=2, log=quiet)
        except RuntimeError:
            assert run.logged == []
            return
        raise AssertionError("an older checkpoint replaced a later one")


def test_upload_deletes_only_older_versions():
    versions = [FakeArtifact(f"ckpt-exp:v{i}", step) for i, step in enumerate([10, 20, 30, 40])]
    versions[1].metadata["step"] = 999  # a later step in an old version is never deleted
    with tempfile.TemporaryDirectory() as d, fake_wandb(latest=versions[-1], versions=versions):
        path = os.path.join(d, "ckpt.pt")
        torch.save({"step": 50}, path)
        wandb_io.upload_checkpoint(FakeRun(), path, "exp", 50, keep=2, log=quiet)
    assert [a.deleted for a in versions] == [True, False, False, False]


# ---------------------------------------------------------------------------- resume checks

def resume_args(**overrides):
    args = dict(experiment="exp", seed=0, train_rows=None, enc_tokens=8192, dec_tokens=12288,
                lr=4e-4, total_steps=100, warmup_steps=10, decay_frac=0.15, min_lr_ratio=0.1)
    args.update(overrides)
    return Namespace(**args)


def resume_ckpt(args, world=2):
    return {"experiment": args.experiment,
            "data": {"world_size": world, **{k: getattr(args, k) for k in DATA_ARGS}},
            "schedule": {k: getattr(args, k) for k in SCHEDULE_ARGS}}


def expect_exit(fn):
    try:
        fn()
    except SystemExit:
        return
    raise AssertionError("expected SystemExit")


def test_resume_takes_data_args_from_checkpoint():
    ck = resume_ckpt(resume_args(dec_tokens=10_000, seed=3))
    args = resume_args()
    check_resume(ck, args, world=2, log=quiet)
    assert args.dec_tokens == 10_000 and args.seed == 3


def test_resume_refuses_schedule_change_and_other_mismatches():
    ck = resume_ckpt(resume_args())
    expect_exit(lambda: check_resume(ck, resume_args(total_steps=200), world=2, log=quiet))
    expect_exit(lambda: check_resume(ck, resume_args(lr=1e-3), world=2, log=quiet))
    expect_exit(lambda: check_resume(ck, resume_args(experiment="other"), world=2, log=quiet))
    expect_exit(lambda: check_resume(ck, resume_args(), world=1, log=quiet))
    old = {k: v for k, v in ck.items() if k != "schedule"}  # written by an older train.py
    expect_exit(lambda: check_resume(old, resume_args(), world=2, log=quiet))


# ---------------------------------------------------------------------------- token cache

def test_build_token_cache_round_trip():
    tok = data.load_tokenizer()
    rows = [(f"question {i}?", f"print({i} * {i})\n") for i in range(23)]
    with tempfile.TemporaryDirectory() as d:
        shard_dir = os.path.join(d, "data", "train")
        os.makedirs(shard_dir)
        for k, part in enumerate([rows[:10], rows[10:]]):  # two shards
            table = pa.table({"question": [q for q, _ in part], "code": [c for _, c in part]})
            pq.write_table(table, os.path.join(shard_dir, f"{k:05d}.parquet"))
        real = data.snapshot_download
        data.snapshot_download = lambda *a, **kw: d
        try:
            data.build_token_cache(os.path.join(d, "all"), None, log=quiet)
            data.build_token_cache(os.path.join(d, "some"), 7, seed=1, log=quiet)
        finally:
            data.snapshot_download = real
        everything = data.TokenizedPairs(os.path.join(d, "all"))
        assert len(everything) == len(rows)
        for i, (q, c) in enumerate(rows):
            assert everything.question(i).tolist() == tok.encode(q, add_special_tokens=False).ids
            assert everything.code(i).tolist() == tok.encode(c, add_special_tokens=False).ids
        sample = data.TokenizedPairs(os.path.join(d, "some"))
        all_pairs = {(everything.question(i).tobytes(), everything.code(i).tobytes()) for i in range(len(rows))}
        assert len(sample) == 7
        assert all((sample.question(i).tobytes(), sample.code(i).tobytes()) in all_pairs for i in range(7))
        assert isinstance(sample.q_flat, np.memmap)  # memory-mapped, shared between processes
        assert not os.path.exists(os.path.join(d, "all.tmp"))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
