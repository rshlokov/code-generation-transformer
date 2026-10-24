"""DDP + FP16 training of the encoder-decoder transformer.

Kaggle (2x T4), first notebook cell:   import os, time; os.environ["SESSION_START"] = str(time.time())
then:                                  torchrun --nproc_per_node=2 train.py --experiment <name>

A run spans several Kaggle sessions. Each session stops cleanly --time_budget_hours after
SESSION_START (the script start if unset), uploads its checkpoint to wandb as the artifact
ckpt-<experiment>:latest, and the next session with the same --experiment downloads it and
continues the same wandb run (id = <experiment>). Within a session a crashed run resumes from the
local <out_dir>/ckpt.pt if that is newer. The token cache comes from wandb when prepare_data.py
(or an earlier session) has stored it, and is built and uploaded otherwise.
"""
import os

# Before torch is imported: every step has a different shape, and expandable segments let the
# CUDA caching allocator reuse freed memory instead of fragmenting it into unusable pieces.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import datetime  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from torch.utils.data import DataLoader, IterableDataset  # noqa: E402

from data import (C_EDGES, PAD_ID, Q_EDGES, BucketBatcher, TokenizedPairs,  # noqa: E402
                  build_token_cache, cache_path)
from models.model import ModelConfig, Seq2SeqTransformer, token_loss  # noqa: E402
from wandb_io import (fetch_checkpoint, fetch_token_cache, upload_checkpoint,  # noqa: E402
                      upload_token_cache, wandb_login)

# Arguments that define the data stream; on resume they are taken from the checkpoint so the
# saved (epoch, batch_idx) position points into the same stream.
DATA_ARGS = ("seed", "train_rows", "enc_tokens", "dec_tokens")
# Arguments that define the learning-rate schedule; a resumed run must use the same values.
SCHEDULE_ARGS = ("lr", "total_steps", "warmup_steps", "decay_frac", "min_lr_ratio")


def parse_args():
    default_out = "/kaggle/working" if os.path.isdir("/kaggle/working") else "runs"
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True,
                   help="wandb run id and checkpoint artifact name ckpt-<experiment>")
    p.add_argument("--out_dir", default=default_out)
    p.add_argument("--train_rows", type=int, default=None,
                   help="random train rows to tokenize (default: all 11.8M)")
    p.add_argument("--enc_tokens", type=int, default=8_192, help="padded encoder tokens per batch per GPU")
    p.add_argument("--dec_tokens", type=int, default=12_288, help="padded decoder tokens per batch per GPU")
    p.add_argument("--total_steps", type=int, default=55_000)
    p.add_argument("--warmup_steps", type=int, default=2_000)
    p.add_argument("--decay_frac", type=float, default=0.15, help="share of total steps used for the final decay")
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--ckpt_minutes", type=float, default=25.0, help="local checkpoint interval")
    p.add_argument("--upload_hours", type=float, default=1.0, help="wandb checkpoint upload interval")
    p.add_argument("--session_start", type=float, default=float(os.environ.get("SESSION_START", time.time())),
                   help="unix time the Kaggle session started (default: $SESSION_START, else now)")
    p.add_argument("--time_budget_hours", type=float, default=11.0,
                   help="clean stop this long after --session_start")
    p.add_argument("--keep_ckpts", type=int, default=2, help="checkpoint versions kept in wandb")
    p.add_argument("--loader_workers", type=int, default=1, help="processes preparing batches ahead")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb_project", default="code-generation-transformer")
    p.add_argument("--wandb_mode", default=None, help="online / offline / disabled")
    return p.parse_args()


def wsd_lambda(warmup, total, decay_frac, min_ratio):
    """Warmup-stable-decay multiplier as a function of the global optimizer step."""
    decay_steps = max(1, round(decay_frac * total))
    decay_start = total - decay_steps

    def f(step):
        if step < warmup:
            return (step + 1) / warmup  # linear warmup from peak/warmup to peak
        if step < decay_start:
            return 1.0
        t = min(1.0, (step - decay_start + 1) / decay_steps)
        return 1.0 - (1.0 - min_ratio) * t

    return f


def param_groups(model, weight_decay):
    """Weight decay on weight matrices only: no decay for norms, biases or embeddings."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.ndim >= 2 and "embed" not in name:
            decay.append(p)
        else:
            no_decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def save_checkpoint(path, **state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-save never leaves a broken checkpoint


def get_rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(s["cuda"])


def check_resume(ck, args, world, log):
    """Validate a checkpoint against this run; data arguments are taken from the checkpoint."""
    if ck["experiment"] != args.experiment:
        raise SystemExit(f"checkpoint belongs to experiment {ck['experiment']!r}, not {args.experiment!r}")
    if any(k not in ck["data"] for k in DATA_ARGS) or "schedule" not in ck:
        raise SystemExit("checkpoint was written by an older train.py; start a new --experiment")
    if ck["data"]["world_size"] != world:
        raise SystemExit(f"checkpoint was trained on {ck['data']['world_size']} processes, not {world}")
    changed = [f"--{k} {getattr(args, k)} (checkpoint: {ck['schedule'][k]})"
               for k in SCHEDULE_ARGS if getattr(args, k) != ck["schedule"][k]]
    if changed:
        raise SystemExit("the learning-rate schedule would change mid-run: " + ", ".join(changed)
                         + "; pass the checkpoint's values or start a new --experiment")
    for k in DATA_ARGS:
        if getattr(args, k) != ck["data"][k]:
            log(f"resume: using {k}={ck['data'][k]} from the checkpoint instead of {getattr(args, k)}")
            setattr(args, k, ck["data"][k])


def probe_memory(ddp_model, model, optimizer, shape, cfg, device):
    """Forward and backward on a synthetic batch of the heaviest bucket shape, so a batch that
    doesn't fit fails in the first minute instead of hours into the run. Returns peak bytes."""
    q_len, c_len, bs = shape
    # AdamW allocates its two moment buffers at the first step; stand in for them until then.
    adam_stub = [torch.empty_like(p) for p in model.parameters() for _ in range(2)] if not optimizer.state else []
    g = torch.Generator().manual_seed(0)  # own generator: the global RNG state is checkpointed
    enc = torch.randint(3, cfg.vocab_size, (bs, q_len), generator=g).to(device)
    dec = torch.randint(3, cfg.vocab_size, (bs, c_len), generator=g).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = ddp_model(enc, enc == PAD_ID, dec)
        loss_sum, n_tok = token_loss(logits, dec)
    del logits  # as in the training step
    (loss_sum / n_tok).backward()
    peak = torch.cuda.max_memory_allocated(device)
    del loss_sum, adam_stub
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    return peak


class BatchStream(IterableDataset):
    """The bucketed stream from a resume position, prepared in a DataLoader worker."""

    def __init__(self, batcher, start_epoch, start_batch):
        self.batcher, self.start_epoch, self.start_batch = batcher, start_epoch, start_batch

    def __iter__(self):
        return self.batcher.iterate(start_epoch=self.start_epoch, start_batch=self.start_batch)


def main():
    args = parse_args()
    distributed = "RANK" in os.environ
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30), device_id=device)
        else:
            device = torch.device("cpu")
            dist.init_process_group("gloo", timeout=datetime.timedelta(minutes=30))
        rank, world = dist.get_rank(), dist.get_world_size()
        # Rank 0 alone downloads or builds the checkpoint and the token cache; the other ranks wait
        # at a CPU barrier with a long timeout, so the 30-minute NCCL timeout only covers training.
        wait_group = dist.new_group(backend="gloo", timeout=datetime.timedelta(hours=3))
    else:
        rank, world = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0
    use_amp = device.type == "cuda"
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")

    def log(msg):
        if is_main:
            print(msg, flush=True)

    def wait_for_rank0():
        if distributed:
            dist.barrier(group=wait_group)

    budget_left = args.time_budget_hours - (time.time() - args.session_start) / 3600
    log(f"time budget: {budget_left:.2f} h left of {args.time_budget_hours} h")

    # ---- checkpoint: rank 0 picks the newest of local and wandb, the other ranks wait
    remote = False
    if is_main:
        import wandb
        logged_in = wandb_login()
        if not logged_in and args.wandb_mode is None:
            print("WANDB_API_KEY not found in Kaggle Secrets, env or ~/.netrc; logging offline")
            args.wandb_mode = "offline"
        remote = logged_in and args.wandb_mode in (None, "online")
        if remote:
            fetch_checkpoint(args.wandb_project, args.experiment, ckpt_path, log)
    wait_for_rank0()
    ck = None
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
        check_resume(ck, args, world, log)
    torch.manual_seed(args.seed)

    # ---- data: rank 0 downloads the token cache from wandb or builds it, the other ranks wait
    cache = cache_path(args.out_dir, args.train_rows, args.seed)
    built_cache = False
    if is_main and not os.path.exists(cache):
        if not (remote and fetch_token_cache(args.wandb_project, cache, args.train_rows, args.seed, log)):
            build_token_cache(cache, args.train_rows, seed=args.seed, log=log)
            built_cache = True
    wait_for_rank0()
    data = TokenizedPairs(cache)
    batcher = BucketBatcher(data, args.enc_tokens, args.dec_tokens, rank=rank, world_size=world, seed=args.seed)
    enc_pad, dec_pad = batcher.padding_fraction()
    log(f"dropped {batcher.n_dropped:,}/{batcher.n_total:,} examples longer than "
        f"{Q_EDGES[-1]} (question) or {C_EDGES[-1]} (code) tokens; "
        f"{len(batcher.buckets)} buckets; padding: encoder {enc_pad:.2%}, decoder {dec_pad:.2%}")

    # ---- model / optimizer / schedule / scaler
    cfg = ModelConfig(max_len=max(Q_EDGES[-1], C_EDGES[-1], 512))
    model = Seq2SeqTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model: {n_params / 1e6:.2f}M parameters, ffn hidden {cfg.ffn_hidden}")
    optimizer = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr,
                                  betas=(0.9, 0.95), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, wsd_lambda(args.warmup_steps, args.total_steps, args.decay_frac, args.min_lr_ratio))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step, epoch, batch_idx = 0, 0, 0
    n_skipped = n_clipped = 0
    pad_totals = torch.zeros(4, dtype=torch.float64)  # enc real, enc slots, dec real, dec slots
    if ck is not None:
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        scheduler.load_state_dict(ck["scheduler"])
        step, epoch, batch_idx = ck["step"], ck["data"]["epoch"], ck["data"]["batch_idx"]
        n_skipped, n_clipped, pad_totals = ck["n_skipped"], ck["n_clipped"], ck["pad_totals"].clone()
        set_rng_state(ck["rng"][rank])
        log(f"resumed from {ckpt_path} at step {step} (epoch {epoch}, batch {batch_idx})")
        del ck
    if step >= args.total_steps:
        log(f"experiment {args.experiment} already finished ({step}/{args.total_steps} steps)")
        if distributed:
            dist.destroy_process_group()
        return

    ddp_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None) if distributed else model

    # ---- fail fast: the heaviest batch must fit before any real step runs
    mem_probe = {}
    if device.type == "cuda":
        shape = batcher.heaviest_shape()
        total = torch.cuda.get_device_properties(device).total_memory
        try:
            peak = probe_memory(ddp_model, model, optimizer, shape, cfg, device)
        except torch.OutOfMemoryError:
            print(f"rank {rank}: the heaviest batch {shape} (question len, code len, batch size) does not "
                  f"fit in GPU memory; lower --dec_tokens / --enc_tokens", flush=True)
            raise
        log(f"memory probe: heaviest batch {shape} peaks at {peak / 2**30:.2f} of {total / 2**30:.2f} GiB")
        if peak > 0.9 * total:
            log("warning: less than 10% GPU memory headroom; fragmentation could still cause an OOM")
        mem_probe = {"mem/probe_peak_gib": peak / 2**30, "mem/probe_shape": list(shape)}

    # Batches are prepared ahead in worker processes, started before wandb's background threads.
    loader = DataLoader(BatchStream(batcher, epoch, batch_idx), batch_size=None,
                        num_workers=args.loader_workers, pin_memory=use_amp,
                        prefetch_factor=4 if args.loader_workers > 0 else None)
    stream = iter(loader)

    # ---- wandb (rank 0 only); the experiment name is the run id, so every session continues one run
    wb = None
    if is_main:
        wb = wandb.init(project=args.wandb_project, id=args.experiment, resume="allow", dir=args.out_dir,
                        mode=args.wandb_mode, config={**vars(args), **cfg.__dict__, "world_size": world,
                                                      "q_edges": Q_EDGES, "c_edges": C_EDGES})
        wb.summary.update({"data/dropped_examples": batcher.n_dropped, "data/total_examples": batcher.n_total,
                           "data/n_buckets": len(batcher.buckets), "data/enc_pad_frac_epoch": enc_pad,
                           "data/dec_pad_frac_epoch": dec_pad, "model/n_params": n_params, **mem_probe})
        if built_cache and remote:
            upload_token_cache(wb, cache, args.train_rows, args.seed)  # in the background; next sessions download it

    saved_step = uploaded_step = -1

    def checkpoint(upload):
        """Collective: every rank contributes its RNG state, rank 0 saves (and uploads).
        Skipped when this step is already saved (and uploaded, if asked): the final checkpoint
        after a step that just hit the upload interval would otherwise upload twice."""
        nonlocal saved_step, uploaded_step
        if saved_step == step and (not upload or uploaded_step == step):
            return
        saved_step = step
        uploaded_step = step if upload else uploaded_step
        rng = [None] * world
        if distributed:
            dist.all_gather_object(rng, get_rng_state())
        else:
            rng[0] = get_rng_state()
        if not is_main:
            return
        save_checkpoint(ckpt_path, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict(), scheduler=scheduler.state_dict(), step=step,
                        data={"epoch": epoch, "batch_idx": batch_idx, "world_size": world,
                              **{k: getattr(args, k) for k in DATA_ARGS}},
                        schedule={k: getattr(args, k) for k in SCHEDULE_ARGS},
                        rng=rng, n_skipped=n_skipped, n_clipped=n_clipped, pad_totals=pad_totals,
                        experiment=args.experiment, model_config=cfg.__dict__)
        log(f"checkpoint saved at step {step}")
        if upload and remote:
            upload_checkpoint(wb, ckpt_path, args.experiment, step, args.keep_ckpts, log)

    # ---- training loop
    last_ckpt = last_upload = time.monotonic()
    ddp_model.train()
    while step < args.total_steps:
        t0 = time.monotonic()
        epoch, b, batch = next(stream)
        batch_idx = b + 1  # resume position: the next batch
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        enc_mask = batch["encoder_padding_mask"]

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = ddp_model(batch["encoder_input_ids"], enc_mask, batch["decoder_input_ids"])
            loss_sum, n_tok = token_loss(logits, batch["labels"])
        del logits  # the loss holds what backward needs; don't keep the fp16 logits alive too

        # Per-token mean over the global batch: DDP averages gradients over ranks, so scale
        # the local sum by world / global_tokens.
        stats = torch.stack([loss_sum.detach().double(), n_tok.double(),
                             (~enc_mask).sum().double(), torch.tensor(enc_mask.numel(), device=device).double(),
                             n_tok.double(), torch.tensor(batch["labels"].numel(), device=device).double()])
        if distributed:
            dist.all_reduce(stats)
        loss = loss_sum * world / stats[1].float()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # clip the real gradients, not the scaled ones
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
        scale_before = scaler.get_scale()
        scaler.step(optimizer)  # skipped by the scaler if gradients contain inf/nan
        scaler.update()
        skipped = scaler.get_scale() < scale_before
        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        n_skipped += int(skipped)
        n_clipped += int(math.isfinite(grad_norm) and grad_norm > args.clip)

        stats = stats.cpu()
        pad_totals += stats[2:]
        if is_main:
            dt = time.monotonic() - t0
            metrics = {
                "train/loss": (stats[0] / stats[1]).item(),
                "train/lr": lr,
                "train/grad_norm": grad_norm,
                "train/clip_rate": n_clipped / step,
                "train/clipped_steps": n_clipped,
                "amp/scale": scaler.get_scale(),
                "amp/skipped_steps": n_skipped,
                "data/epoch": epoch,
                "data/enc_pad_frac": 1 - (stats[2] / stats[3]).item(),
                "data/dec_pad_frac": 1 - (stats[4] / stats[5]).item(),
                "data/enc_pad_frac_running": 1 - (pad_totals[0] / pad_totals[1]).item(),
                "data/dec_pad_frac_running": 1 - (pad_totals[2] / pad_totals[3]).item(),
                "data/batch_size": batch["labels"].size(0),
                "perf/tokens_per_s": (stats[3] + stats[5]).item() / dt,
            }
            if use_amp:
                metrics["mem/peak_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            wb.log(metrics, step=step)
            print(f"step {step:5d} | loss {metrics['train/loss']:.4f} | lr {lr:.2e} | "
                  f"gnorm {grad_norm:.3f} | scale {metrics['amp/scale']:.0f} | skipped {n_skipped} | "
                  f"{dt * 1000:.0f} ms", flush=True)

        # Rank 0 owns the clock and broadcasts (save, upload, stop), so all ranks act on the same step.
        now = time.monotonic()
        flags = torch.tensor([now - last_ckpt >= args.ckpt_minutes * 60,
                              now - last_upload >= args.upload_hours * 3600,
                              time.time() - args.session_start >= args.time_budget_hours * 3600],
                             dtype=torch.int32, device=device)
        if distributed:
            dist.broadcast(flags, src=0)
        save, upload, stop = flags.tolist()
        if stop:
            log(f"time budget of {args.time_budget_hours} h reached at step {step}; stopping")
            break
        if save or upload:
            checkpoint(upload)
            last_ckpt = time.monotonic()
            last_upload = last_ckpt if upload else last_upload

    checkpoint(upload=True)
    if is_main:
        wb.summary.update({"data/enc_pad_frac_seen": 1 - (pad_totals[0] / pad_totals[1]).item(),
                           "data/dec_pad_frac_seen": 1 - (pad_totals[2] / pad_totals[3]).item()})
        wb.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
