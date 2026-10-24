"""Greedy generation for the 1,919 test questions -> predictions.jsonl.

python predict.py --ckpt /kaggle/working/ckpt.pt                # KV cache, 170 s deadline
python predict.py --experiment <name>                           # newest of local ckpt and wandb ckpt-<name>:latest
python predict.py --ckpt /kaggle/working/ckpt.pt --benchmark    # time cached vs uncached on one GPU
"""
import os

# Before torch is imported (see train.py): batches shrink as rows finish, so shapes vary.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from data import C_EDGES, EOS_ID, PAD_ID, Q_EDGES, REPO, load_tokenizer  # noqa: E402
from models.model import ModelConfig, Seq2SeqTransformer, greedy_generate  # noqa: E402
from wandb_io import fetch_checkpoint, wandb_login  # noqa: E402

N_TEST = 1_919


def parse_args():
    default_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "runs"
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(default_dir, "ckpt.pt"))
    p.add_argument("--experiment", default=None,
                   help="download ckpt-<experiment>:latest from wandb into --ckpt if it is newer")
    p.add_argument("--out", default=os.path.join(default_dir, "predictions.jsonl"))
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=C_EDGES[-1] - 1,
                   help="longest code seen in training (the last code bucket holds <bos> + code)")
    p.add_argument("--deadline", type=float, default=170.0, help="seconds for the whole generation")
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--no_compact", action="store_true", help="keep finished rows in the batch")
    p.add_argument("--benchmark", action="store_true",
                   help="time cached and uncached generation for all questions (no deadline)")
    p.add_argument("--wandb_project", default="code-generation-transformer")
    p.add_argument("--wandb_mode", default=None)
    return p.parse_args()


def load_test_questions():
    path = hf_hub_download(REPO, "data/test/test.parquet", repo_type="dataset")
    return pq.read_table(path, columns=["question"]).column("question").to_pylist()


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = Seq2SeqTransformer(ModelConfig(**ck["model_config"]))
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()  # weights stay fp32: generate_all runs under autocast, as training does
    return model, ck.get("experiment", ck.get("wandb_id"))


def generate_all(model, questions, tok, device, batch_size, max_new_tokens, use_cache=True,
                 deadline_s=None, compact=True):
    """Programs for every question, in the input order. Never raises for a failed batch:
    its questions get an empty program. Returns (programs, stats).

    Runs under fp16 autocast on GPU, the same numerics as training: the residual stream stays
    fp32 and the matmuls run in fp16 (model.half() would also put the residual stream in fp16).
    """
    t_start = time.monotonic()
    deadline = t_start + deadline_s if deadline_s else None
    max_q = Q_EDGES[-1] - 1  # training drops longer questions; the last bucket holds question + <eos>
    ids = [e.ids[:max_q] + [EOS_ID] for e in tok.encode_batch(questions, add_special_tokens=False)]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))  # similar lengths finish together
    programs = [""] * len(ids)
    n_failed_batches = n_skipped = 0
    batch_seconds = []
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda")
    for s in range(0, len(order), batch_size):
        chunk = order[s:s + batch_size]
        if deadline is not None and time.monotonic() > deadline:
            n_skipped += len(chunk)
            continue
        try:
            t_batch = time.monotonic()
            L = max(len(ids[i]) for i in chunk)
            enc = torch.full((len(chunk), L), PAD_ID, dtype=torch.long)
            for j, i in enumerate(chunk):
                enc[j, :len(ids[i])] = torch.tensor(ids[i])
            enc = enc.to(device)
            with autocast:
                out = greedy_generate(model, enc, enc == PAD_ID, max_new_tokens, use_cache=use_cache,
                                      deadline=deadline, compact=compact)
            for i, toks in zip(chunk, out):
                programs[i] = tok.decode(toks, skip_special_tokens=True)
            batch_seconds.append(time.monotonic() - t_batch)  # greedy_generate's .tolist() already synced
        except Exception as e:  # keep going: every question must still get a line
            print(f"batch at {s} failed: {e!r}")
            n_failed_batches += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    stats = {"seconds": time.monotonic() - t_start, "questions_skipped_by_deadline": n_skipped,
             "failed_batches": n_failed_batches,
             "deadline_hit": deadline is not None and time.monotonic() > deadline,
             "batches": len(batch_seconds),
             "batch_seconds_mean": sum(batch_seconds) / max(1, len(batch_seconds)),
             "batch_seconds_max": max(batch_seconds, default=0.0)}
    return programs, stats


def write_predictions(path, programs, n_expected=N_TEST):
    programs = list(programs)[:n_expected]
    programs += [""] * (n_expected - len(programs))
    with open(path, "w", encoding="utf-8") as f:
        for code in programs:
            f.write(json.dumps({"code": code}) + "\n")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    questions = load_test_questions()
    assert len(questions) == N_TEST, len(questions)
    programs, report, experiment = [""] * N_TEST, {}, None
    import wandb
    logged_in = wandb_login()
    if not logged_in and args.wandb_mode is None:
        args.wandb_mode = "offline"
    try:
        if args.experiment and logged_in and args.wandb_mode in (None, "online"):
            os.makedirs(os.path.dirname(os.path.abspath(args.ckpt)), exist_ok=True)
            fetch_checkpoint(args.wandb_project, args.experiment, args.ckpt, print)
        tok = load_tokenizer()
        model, experiment = load_model(args.ckpt, device)
        kw = dict(batch_size=args.batch_size, max_new_tokens=args.max_new_tokens, compact=not args.no_compact)
        if args.benchmark:
            for name, cached in [("cached", True), ("uncached", False)]:
                _, st = generate_all(model, questions, tok, device, use_cache=cached, **kw)
                report[f"predict/time_{name}_s"] = st["seconds"]
                print(f"{name}: {st['seconds']:.1f} s for {N_TEST} questions in {st['batches']} batches of "
                      f"{args.batch_size}; per batch mean {st['batch_seconds_mean']:.2f} s, "
                      f"max {st['batch_seconds_max']:.2f} s")
        programs, st = generate_all(model, questions, tok, device, use_cache=not args.no_cache,
                                    deadline_s=args.deadline, **kw)
        report.update({f"predict/{k}": v for k, v in st.items()})
        print(report)
    finally:
        write_predictions(args.out, programs)  # always exactly N_TEST lines
        print(f"wrote {args.out}")

    # A run of its own, grouped with the training run: reopening the training run by id would
    # clash with a training session that is still writing to it.
    run = wandb.init(project=args.wandb_project, group=experiment, job_type="predict",
                     name=f"{experiment}-predict" if experiment else None, mode=args.wandb_mode,
                     dir=os.path.dirname(os.path.abspath(args.out)), config=vars(args))
    run.summary.update({**report, "predict/device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu"})
    run.finish()


if __name__ == "__main__":
    main()
