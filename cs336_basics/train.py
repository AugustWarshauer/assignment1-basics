"""Configurable training entry point for the CS336 assignment 1 experiments.

The notebook (`training_loop.ipynb`) is for data prep and inspection; this module
holds the training loop itself so that a sweep is a loop over `TrainConfig`s
rather than a hand-edited cell. Run a single config from the shell with:

    uv run python -m cs336_basics.train --run-name baseline --max-iters 5000

or from the notebook with:

    from cs336_basics.train import TrainConfig, train
    train(TrainConfig(run_name="baseline", max_iters=5000))
"""

import argparse
import contextlib
import dataclasses
import itertools
import json
import math
import os
import pickle
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from cs336_basics.training import (
    AdamW,
    cross_entropy,
    gradient_clipping,
    learning_rate_schedule,
    load_checkpoint,
    save_checkpoint,
)
from cs336_basics.transformer import Transformer_LM

#: Repo root, so every path works regardless of the cwd a notebook was started from.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainConfig:
    """Everything that defines one run. Logged verbatim to wandb as the run config."""

    run_name: str = "baseline"
    seed: int = 0

    # --- data ---
    train_tokens: Path = DATA / "owt_train_tokens.bin"
    valid_tokens: Path = DATA / "owt_valid_tokens.bin"

    # --- model ---
    vocab_size: int = 32000
    context_length: int = 256
    num_layers: int = 4
    d_model: int = 512
    num_heads: int = 16
    d_ff: int = 1344
    theta: float = 10000.0

    # --- optimization ---
    batch_size: int = 32
    max_lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 500
    max_iters: int = 5000
    total_tokens: int | None = None  # if set, overrides max_iters = total_tokens / (batch_size * context_length)
    max_grad_norm: float = 1.0
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)

    # --- evaluation / checkpointing ---
    eval_interval: int = 250
    eval_iters: int = 20
    eval_batch_size: int | None = None  # defaults to batch_size; fix it to compare runs of different batch sizes
    checkpoint_dir: Path = ROOT / "checkpoints"
    resume_from: Path | None = None
    keep_only_latest_and_best: bool = True
    save_checkpoints: bool = True  # off for short timed trials, where writing ~600MB per eval eats the time budget

    # --- logging ---
    device: str = field(default_factory=pick_device)
    wandb_project: str = "cs336-owt"
    wandb_mode: str = "online"  # "offline" or "disabled" to skip the network
    compile: bool = False
    autocast: bool = False  # bf16 mixed precision for forward passes; the loss is computed in fp32
    # If set, the run lasts this many wall-clock minutes (the leaderboard rule): the cosine schedule and its warmup
    # (warmup_frac of the budget) run on elapsed time, there are 20 evals, and max_iters / total_tokens are ignored.
    time_budget_minutes: float | None = None
    warmup_frac: float = 0.02

    @property
    def run_dir(self) -> Path:
        return Path(self.checkpoint_dir) / self.run_name


def set_seed(seed: int) -> None:
    """Seed every RNG the run touches.

    The ablations in section 7.3 are single-variable comparisons against a
    baseline, so the two runs have to differ *only* in the thing being ablated —
    otherwise part of the gap you measure is just a different weight init and a
    different batch order.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _batch_from_starts(data: np.ndarray, starts: np.ndarray, context_length: int, device: str):
    """Build one (inputs, targets) batch from explicit start offsets.

    `training.data_loading` samples its own random offsets, which is what we want
    for training. For evaluation we want the *same* batches every time so the
    validation curve reflects learning rather than which windows happened to be
    drawn, hence this variant.
    """
    inputs = np.stack([data[i : i + context_length] for i in starts])
    targets = np.stack([data[i + 1 : i + context_length + 1] for i in starts])
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def make_eval_starts(data: np.ndarray, cfg: TrainConfig, seed_offset: int) -> np.ndarray:
    """Pick a fixed set of evaluation windows once, from a dedicated RNG.

    Uses its own Generator so that drawing eval batches never perturbs the global
    numpy stream that the training data loader consumes.
    """
    rng = np.random.default_rng(cfg.seed + seed_offset)
    return rng.integers(0, len(data) - cfg.context_length - 1, size=cfg.eval_iters * (cfg.eval_batch_size or cfg.batch_size))


def _amp(cfg: TrainConfig):
    return torch.autocast(cfg.device.split(":")[0], dtype=torch.bfloat16) if cfg.autocast else contextlib.nullcontext()


@torch.no_grad()
def estimate_loss(model, splits: dict[str, tuple[np.ndarray, np.ndarray]], cfg: TrainConfig) -> dict[str, float]:
    """Average loss over the fixed evaluation batches for each split."""
    model.eval()
    out: dict[str, float] = {}
    for name, (data, starts) in splits.items():
        total = 0.0
        for i in range(cfg.eval_iters):
            eval_bs = cfg.eval_batch_size or cfg.batch_size
            chunk = starts[i * eval_bs : (i + 1) * eval_bs]
            x, y = _batch_from_starts(data, chunk, cfg.context_length, cfg.device)
            with _amp(cfg):
                logits = model(x)
            total += cross_entropy(logits.float(), y).item()
        out[name] = total / cfg.eval_iters
    model.train()
    return out


def train(cfg: TrainConfig):
    """Run one training configuration end to end. Returns the trained model."""
    import wandb

    if cfg.total_tokens is not None:
        cfg = dataclasses.replace(cfg, max_iters=cfg.total_tokens // (cfg.batch_size * cfg.context_length))
    if cfg.device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")  # TF32; the handout warns not to do this on mps
    set_seed(cfg.seed)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / "config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(cfg).items()}, indent=1))
    if cfg.resume_from is None and (cfg.run_dir / "latest.pt").exists():
        cfg = dataclasses.replace(cfg, resume_from=cfg.run_dir / "latest.pt")  # reusing a run name continues that run

    # --- data (memmap: pages are faulted in lazily, never loaded whole) ---
    train_data = np.memmap(cfg.train_tokens, dtype=np.uint16, mode="r")
    valid_data = np.memmap(cfg.valid_tokens, dtype=np.uint16, mode="r")
    print(f"train tokens: {len(train_data):,}   valid tokens: {len(valid_data):,}")

    splits = {
        "train": (train_data, make_eval_starts(train_data, cfg, seed_offset=1)),
        "val": (valid_data, make_eval_starts(valid_data, cfg, seed_offset=2)),
    }

    # --- model / optimizer ---
    model = Transformer_LM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        num_layers=cfg.num_layers,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.context_length,
        theta=cfg.theta,
        device=cfg.device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {cfg.device}   parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=cfg.max_lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    # checkpoints are saved from `model`: a compiled module's state_dict keys gain an "_orig_mod." prefix
    fwd = torch.compile(model) if cfg.compile else model
    diverge_at = 2 * math.log(cfg.vocab_size)

    start_iter = 1
    best_val = float("inf")
    wandb_id = None
    elapsed0 = 0.0
    if cfg.resume_from is not None:
        start_iter = load_checkpoint(cfg.resume_from, model, optimizer) + 1
        state_path = Path(cfg.resume_from).with_suffix(".state.pkl")
        if state_path.exists():
            with open(state_path, "rb") as f:
                state = pickle.load(f)
            best_val, wandb_id, elapsed0 = state["best_val"], state["wandb_id"], state.get("elapsed", 0.0)
            np.random.set_state(state["np_rng"])
        print(f"resumed from {cfg.resume_from} at iteration {start_iter}")

    wandb_id = wandb_id or uuid.uuid4().hex[:12]
    wandb.init(
        project=cfg.wandb_project,
        name=cfg.run_name,
        id=wandb_id,
        resume="allow",
        mode=cfg.wandb_mode,
        config={
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in dataclasses.asdict(cfg).items()},
            "n_params": n_params,
        },
    )

    def checkpoint(path: Path):
        save_checkpoint(model, optimizer, t, path)
        with open(path.with_suffix(".state.pkl"), "wb") as f:
            pickle.dump({"best_val": best_val, "np_rng": np.random.get_state(), "wandb_id": wandb_id,
                         "elapsed": time.perf_counter() - t0}, f)

    budget = cfg.time_budget_minutes * 60 if cfg.time_budget_minutes else None
    next_eval = (math.floor(elapsed0 / (budget / 20)) + 1) * budget / 20 if budget else None
    steps = itertools.count(start_iter) if budget else range(start_iter, cfg.max_iters + 1)
    t = start_iter - 1
    t0 = time.perf_counter() - elapsed0
    try:
        for t in tqdm(steps, initial=start_iter, total=None if budget else cfg.max_iters):
            if budget:
                frac = min((time.perf_counter() - t0) / budget, 1.0)
                lr = learning_rate_schedule(frac, cfg.max_lr, cfg.min_lr, cfg.warmup_frac, 1.0)
            else:
                lr = learning_rate_schedule(t, cfg.max_lr, cfg.min_lr, cfg.warmup_iters, cfg.max_iters)
            for group in optimizer.param_groups:
                group["lr"] = lr

            x, y = _batch_from_starts(
                train_data,
                np.random.randint(0, len(train_data) - cfg.context_length - 1, size=cfg.batch_size),
                cfg.context_length,
                cfg.device,
            )
            with _amp(cfg):
                logits = fwd(x)
            loss = cross_entropy(logits.float(), y)

            optimizer.zero_grad()
            loss.backward()
            gradient_clipping(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            loss_val = loss.item()
            elapsed = time.perf_counter() - t0
            progress = {"wallclock_s": elapsed, "tokens_seen": t * cfg.batch_size * cfg.context_length}
            wandb.log({"train/loss": loss_val, "lr": lr, **progress}, step=t)
            if not math.isfinite(loss_val) or loss_val > diverge_at:
                print(f"diverged at step {t}: loss {loss_val}")
                wandb.run.summary["diverged_at_step"] = t
                (cfg.run_dir / "diverged.txt").write_text(f"{t}\n")
                break

            out_of_time = budget is not None and elapsed >= budget
            if budget is None:
                eval_now = t % cfg.eval_interval == 0 or t == cfg.max_iters
            else:
                eval_now = elapsed >= next_eval or out_of_time
                while next_eval <= elapsed:
                    next_eval += budget / 20
            if eval_now:
                losses = estimate_loss(fwd, splits, cfg)
                print(f"step {t}: train {losses['train']:.4f}  val {losses['val']:.4f}  lr {lr:.6f}")
                wandb.log({"val/loss": losses["val"], "eval/train_loss": losses["train"], **progress}, step=t)

                # Keep only `latest` and `best` — each checkpoint carries Adam's m and v,
                # so a full history is roughly 3x model size per eval.
                improved = losses["val"] < best_val
                if improved:
                    best_val = losses["val"]
                if cfg.save_checkpoints:
                    if improved:
                        checkpoint(cfg.run_dir / "best.pt")
                    checkpoint(cfg.run_dir / "latest.pt")
                    if not cfg.keep_only_latest_and_best:
                        checkpoint(cfg.run_dir / f"step_{t}.pt")
            if out_of_time:
                break
    finally:
        # Runs even if training raises or the kernel is interrupted, so the run is
        # closed out properly and the next wandb.init() starts clean.
        wandb.finish()

    loop_seconds = time.perf_counter() - t0
    steps_run = t - start_iter + 1
    summary = {"last_step": t, "steps_run": steps_run, "loop_seconds": round(loop_seconds, 1), "best_val": best_val,
               "tokens_per_second": round(t * cfg.batch_size * cfg.context_length / max(loop_seconds, 1e-9))}  # whole run, across resumes
    (cfg.run_dir / "summary.json").write_text(json.dumps(summary))
    print(f"done in {loop_seconds / 60:.1f} min   best val loss {best_val:.4f}   {summary['tokens_per_second']:,} tokens/s")
    return model


def _parse_args() -> TrainConfig:
    """Expose every TrainConfig field as a --flag, so sweeps can be driven from a shell."""
    parser = argparse.ArgumentParser(description=__doc__)
    for f in dataclasses.fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        default = getattr(TrainConfig(), f.name)
        if f.type is bool or isinstance(default, bool):
            parser.add_argument(flag, type=lambda s: s.lower() in {"1", "true", "yes"}, default=default)
        elif f.name == "betas":
            parser.add_argument(flag, type=float, nargs=2, default=default)
        elif "Path" in str(f.type):
            parser.add_argument(flag, type=Path, default=default)
        else:
            parser.add_argument(flag, type=type(default) if default is not None else str, default=default)
    args = parser.parse_args()
    kwargs = {f.name: getattr(args, f.name) for f in dataclasses.fields(TrainConfig)}
    if isinstance(kwargs["betas"], list):
        kwargs["betas"] = tuple(kwargs["betas"])
    return TrainConfig(**kwargs)


if __name__ == "__main__":
    train(_parse_args())
