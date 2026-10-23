"""DDP + FP16 training of the encoder-decoder transformer.

Kaggle (2x T4):  torchrun --nproc_per_node=2 train.py
Resumes automatically from <out_dir>/ckpt.pt if it exists.
"""
import argparse
import math
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data import C_EDGES, Q_EDGES, BucketBatcher, TokenizedPairs, build_token_cache
from models.model import ModelConfig, Seq2SeqTransformer, token_loss


def parse_args():
    default_out = "/kaggle/working" if os.path.isdir("/kaggle/working") else "runs"
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=default_out)
    p.add_argument("--train_rows", type=int, default=500_000, help="random train rows to tokenize")
    p.add_argument("--token_budget", type=int, default=24_576, help="encoder+decoder tokens per batch per GPU")
    p.add_argument("--total_steps", type=int, default=200)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--decay_frac", type=float, default=0.15, help="share of total steps used for the final decay")
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--ckpt_minutes", type=float, default=2.0)
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


def wandb_login():
    """API key from Kaggle Secrets, falling back to the WANDB_API_KEY env variable."""
    import wandb
    key = None
    try:
        from kaggle_secrets import UserSecretsClient
        key = UserSecretsClient().get_secret("WANDB_API_KEY")
    except Exception:
        key = os.environ.get("WANDB_API_KEY")
    if key:
        wandb.login(key=key)
    return key is not None


def save_checkpoint(path, **state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-save never leaves a broken checkpoint


def main():
    args = parse_args()
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank, world = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0
    use_amp = device.type == "cuda"
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")

    def log(msg):
        if is_main:
            print(msg, flush=True)

    # ---- data: rank 0 tokenizes once, the other ranks wait and load the cache
    cache = os.path.join(args.out_dir, f"tokens_{args.train_rows}_seed{args.seed}.npz")
    if is_main and not os.path.exists(cache):
        build_token_cache(cache, args.train_rows, seed=args.seed, log=log)
    if distributed:
        dist.barrier()
    data = TokenizedPairs(cache)
    batcher = BucketBatcher(data, args.token_budget, rank=rank, world_size=world, seed=args.seed)
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

    step, epoch, batch_idx, wandb_id = 0, 0, 0, None
    n_skipped = n_clipped = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        scheduler.load_state_dict(ck["scheduler"])
        step, epoch, batch_idx = ck["step"], ck["epoch"], ck["batch_idx"]
        n_skipped, n_clipped, wandb_id = ck["n_skipped"], ck["n_clipped"], ck["wandb_id"]
        log(f"resumed from {ckpt_path} at step {step}")

    ddp_model = DDP(model, device_ids=[device.index]) if distributed else model

    # ---- wandb (rank 0 only); the run id in the checkpoint continues the same run
    wb = None
    if is_main:
        import wandb
        if not wandb_login() and args.wandb_mode is None:
            print("WANDB_API_KEY not found in Kaggle Secrets or env; logging offline")
            args.wandb_mode = "offline"
        wb = wandb.init(project=args.wandb_project, id=wandb_id, resume="allow",
                        mode=args.wandb_mode, config={**vars(args), **cfg.__dict__, "world_size": world,
                                                      "q_edges": Q_EDGES, "c_edges": C_EDGES})
        wandb_id = wb.id
        wb.summary.update({"data/dropped_examples": batcher.n_dropped, "data/total_examples": batcher.n_total,
                           "data/n_buckets": len(batcher.buckets), "data/enc_pad_frac_epoch": enc_pad,
                           "data/dec_pad_frac_epoch": dec_pad, "model/n_params": n_params})

    def checkpoint():
        save_checkpoint(ckpt_path, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict(), scheduler=scheduler.state_dict(), step=step,
                        epoch=epoch, batch_idx=batch_idx, n_skipped=n_skipped, n_clipped=n_clipped,
                        wandb_id=wandb_id, model_config=cfg.__dict__)
        log(f"checkpoint saved at step {step}")

    # ---- training loop
    stream = batcher.iterate(start_epoch=epoch, start_batch=batch_idx)
    pad_totals = torch.zeros(4, dtype=torch.float64)  # enc real, enc slots, dec real, dec slots
    last_ckpt = time.monotonic()
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
            wb.log(metrics, step=step)
            print(f"step {step:4d} | loss {metrics['train/loss']:.4f} | lr {lr:.2e} | "
                  f"gnorm {grad_norm:.3f} | scale {metrics['amp/scale']:.0f} | skipped {n_skipped} | "
                  f"{dt * 1000:.0f} ms", flush=True)
            if time.monotonic() - last_ckpt >= args.ckpt_minutes * 60:
                checkpoint()
                last_ckpt = time.monotonic()

    if is_main:
        checkpoint()
        wb.summary.update({"data/enc_pad_frac_seen": 1 - (pad_totals[0] / pad_totals[1]).item(),
                           "data/dec_pad_frac_seen": 1 - (pad_totals[2] / pad_totals[3]).item()})
        wb.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
