"""Tokenize the train split once and store the token cache in wandb.

Training sessions then download the cache (~2.5 GB, a few minutes) instead of tokenizing for
~30 minutes on GPU time. This script needs no GPU, so it can run in a CPU-only Kaggle session.

python prepare_data.py                      # all 11.8M rows, seed 0 (train.py's defaults)
python prepare_data.py --train_rows 500000  # must match train.py's --train_rows / --seed
"""
import argparse
import os

from data import build_token_cache, cache_path
from wandb_io import cache_artifact_name, latest_artifact, upload_token_cache, wandb_login


def main():
    default_out = "/kaggle/working" if os.path.isdir("/kaggle/working") else "runs"
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=default_out)
    p.add_argument("--train_rows", type=int, default=None, help="random train rows to tokenize (default: all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb_project", default="code-generation-transformer")
    args = p.parse_args()

    import wandb
    if not wandb_login():
        raise SystemExit("WANDB_API_KEY not found in Kaggle Secrets, env or ~/.netrc")
    name = cache_artifact_name(args.train_rows, args.seed)
    if latest_artifact(args.wandb_project, name, "dataset") is not None:
        print(f"{name} is already in wandb; nothing to do")
        return
    os.makedirs(args.out_dir, exist_ok=True)
    path = cache_path(args.out_dir, args.train_rows, args.seed)
    if not os.path.exists(path):
        build_token_cache(path, args.train_rows, seed=args.seed)
    run = wandb.init(project=args.wandb_project, job_type="prepare-data", dir=args.out_dir,
                     config=vars(args))
    upload_token_cache(run, path, args.train_rows, args.seed, wait=True)
    run.finish()
    print(f"uploaded {path} as {name}")


if __name__ == "__main__":
    main()
