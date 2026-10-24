"""wandb storage for checkpoints and the token cache (used by train.py, predict.py, prepare_data.py).

Artifacts:
  ckpt-<experiment>            type "model":   ckpt.pt, alias "latest", metadata {"step": ...}
  tokens-<rows|all>-seed<s>    type "dataset": the token cache directory written by data.build_token_cache
"""
import os

import torch


def wandb_login():
    """API key from Kaggle Secrets, falling back to the WANDB_API_KEY env variable or ~/.netrc."""
    import wandb
    key = None
    try:
        from kaggle_secrets import UserSecretsClient
        key = UserSecretsClient().get_secret("WANDB_API_KEY")
    except Exception:
        key = os.environ.get("WANDB_API_KEY")
    if key:
        return wandb.login(key=key)
    return wandb.login(verify=False) if os.path.exists(os.path.expanduser("~/.netrc")) else False


def is_not_found(err):
    """wandb raises CommError both for a missing artifact or project and for network or server
    failures; only the first means that nothing is stored yet."""
    return "not found" in str(err)


def latest_artifact(project, name, type_):
    """<project>/<name>:latest, or None if it doesn't exist. Any other error propagates, so a
    network failure is never mistaken for "no checkpoint" (which would restart training)."""
    import wandb
    try:
        return wandb.Api().artifact(f"{project}/{name}:latest", type=type_)
    except wandb.errors.CommError as e:
        if is_not_found(e):
            return None
        raise


def checkpoint_step(path):
    """Step of a local checkpoint, or -1 if there is none. mmap reads it without loading the tensors."""
    if not os.path.exists(path):
        return -1
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)["step"]


def fetch_checkpoint(project, experiment, ckpt_path, log):
    """Make ckpt_path the newest of the local checkpoint and ckpt-<experiment>:latest."""
    art = latest_artifact(project, f"ckpt-{experiment}", "model")
    if art is None:
        log(f"no checkpoint ckpt-{experiment} in wandb")
        return
    local_step, remote_step = checkpoint_step(ckpt_path), art.metadata.get("step", -1)
    if remote_step <= local_step:
        log(f"local checkpoint (step {local_step}) is not older than wandb {art.name} (step {remote_step})")
        return
    folder = art.download(root=os.path.join(os.path.dirname(ckpt_path), "wandb_ckpt"), skip_cache=True)
    os.replace(os.path.join(folder, "ckpt.pt"), ckpt_path)
    log(f"downloaded {art.name} (step {remote_step})")


def upload_checkpoint(run, ckpt_path, experiment, step, keep, log):
    """Log ckpt_path as ckpt-<experiment>:latest and delete older versions beyond the newest `keep`.

    Refuses to upload when wandb already holds a later step (e.g. two sessions of one experiment),
    and never deletes a version with a later step than this one.
    """
    import wandb
    name, project = f"ckpt-{experiment}", f"{run.entity}/{run.project}"
    remote = latest_artifact(project, name, "model")
    if remote is not None and remote.metadata.get("step", -1) > step:
        raise RuntimeError(f"wandb {remote.name} is at step {remote.metadata['step']}, later than this "
                           f"checkpoint (step {step}); refusing to replace it")
    art = wandb.Artifact(name, type="model", metadata={"step": step})
    art.add_file(ckpt_path, name="ckpt.pt")  # copied to wandb staging, so later saves don't race the upload
    run.log_artifact(art, aliases=["latest"]).wait()
    versions = sorted(wandb.Api().artifacts("model", f"{project}/{name}"),
                      key=lambda a: int(a.version.lstrip("v")), reverse=True)
    old = [a for a in versions[keep:] if a.metadata.get("step", -1) <= step]
    for a in old:
        a.delete(delete_aliases=True)
    log(f"uploaded checkpoint at step {step} as {art.name}; deleted {len(old)} old versions")


def cache_artifact_name(n_rows, seed):
    return f"tokens-{n_rows or 'all'}-seed{seed}"


def fetch_token_cache(project, path, n_rows, seed, log):
    """Download the token cache artifact into path. Returns False if it isn't in wandb yet."""
    art = latest_artifact(project, cache_artifact_name(n_rows, seed), "dataset")
    if art is None:
        return False
    tmp = path + ".download"
    art.download(root=tmp, skip_cache=True)
    os.replace(tmp, path)  # atomic: a half-downloaded cache is never picked up
    log(f"downloaded token cache {art.name} -> {path}")
    return True


def upload_token_cache(run, path, n_rows, seed, wait=False):
    """Log the token cache directory as an artifact; wandb skips the upload if the content exists."""
    import wandb
    art = wandb.Artifact(cache_artifact_name(n_rows, seed), type="dataset",
                         metadata={"train_rows": n_rows, "seed": seed})
    art.add_dir(path, policy="immutable")  # no staging copy: the cache never changes
    logged = run.log_artifact(art)
    if wait:
        logged.wait()
