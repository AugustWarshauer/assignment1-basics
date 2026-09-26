"""Modal launcher; experiments are defined in training_loop.ipynb, whose launch() deploys this app and spawns train_run.

    uv run modal run cs336_basics/modal_app.py::tokenize_corpus --name ts_train      # {ts,owt}_{train,valid} -> data/<name>.bin
    uv run modal deploy cs336_basics/modal_app.py                                     # then spawn train_run from Python

Runs are spawned on the deployed app rather than started from a local entrypoint: a spawned call finishes on Modal even
if the laptop sleeps, whereas `modal run --detach` cancelled .map() inputs when the local client died.
"""
import json
import shutil
from pathlib import Path

import modal

REPO, VOL = Path(__file__).parent.parent, Path("/vol")
app = modal.App("cs336-a1")
vol = modal.Volume.from_name("cs336-a1", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("torch~=2.8.0", "numpy==2.3.2", "einops==0.8.1", "regex==2025.7.34", "tqdm==4.67.1", "wandb==0.21.2")
    # torch 2.8 is the first default PyPI wheel with B200 (sm_100) kernels. cs336_basics/__init__.py reads its version
    # from installed-package metadata, which a copied source tree lacks, hence the stub METADATA file.
    .run_commands("d=$(python -c 'import site; print(site.getsitepackages()[0])')/cs336_basics-1.0.6.dist-info && mkdir -p $d"
                  " && printf 'Metadata-Version: 2.1\\nName: cs336_basics\\nVersion: 1.0.6\\n' > $d/METADATA")
    .add_local_python_source("cs336_basics")
    .add_local_dir(REPO / "data" / "tinystories_bpe_10k", "/root/tokenizers/tinystories_bpe_10k")
    .add_local_dir(REPO / "data" / "owt_bpe_32k", "/root/tokenizers/owt_bpe_32k")
)


@app.function(image=image, volumes={VOL: vol}, cpu=16, memory=16 * 1024, timeout=2 * 3600)
def tokenize_corpus(name: str):
    from cs336_basics.claude_suggested_speedups import download_text, tokenize_file

    ts, split = name.startswith("ts_"), name.split("_")[1]
    url = (f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-{split}.txt" if ts
           else f"https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_{split}.txt.gz")
    tmp = VOL / "data" / f"{name}.tmp"
    tmp.parent.mkdir(exist_ok=True)
    tokenize_file(download_text(url, Path("/tmp")), Path("/root/tokenizers") / ("tinystories_bpe_10k" if ts else "owt_bpe_32k"), tmp, 16)
    tmp.rename(tmp.with_suffix(".bin"))
    vol.commit()


@app.function(image=image, gpu="H100!", volumes={VOL: vol}, secrets=[modal.Secret.from_name("wandb")], timeout=4 * 3600,
              retries=modal.Retries(max_retries=3, initial_delay=0.0), single_use_containers=True, max_containers=4)
def train_run(overrides: dict) -> dict:
    import torch

    from cs336_basics.train import TrainConfig, train

    for key in ("train_tokens", "valid_tokens"):  # copy off the network Volume: the data loader does random reads
        if not (Path("/tmp") / overrides[key]).exists():
            shutil.copy(VOL / "data" / overrides[key], Path("/tmp") / overrides[key])
        overrides[key] = Path("/tmp") / overrides[key]
    cfg = TrainConfig(**overrides, checkpoint_dir=VOL / "checkpoints", device="cuda")
    try:
        train(cfg)
    except torch.cuda.OutOfMemoryError:  # caught so a batch-size sweep reports its memory limit instead of retrying
        return {"run_name": cfg.run_name, "oom": True}
    finally:
        vol.commit()
    return {"run_name": cfg.run_name, **json.loads((cfg.run_dir / "summary.json").read_text())}
