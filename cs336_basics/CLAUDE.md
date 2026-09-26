# CS336 Assignment 1 — Basics

## How to work with me on this repo

**This is a learning exercise, not a delivery task.** The user is taking Stanford
CS336 (Language Modeling from Scratch) and is implementing everything by hand.

### THE RULE: one edit, one question, every time

**NEVER modify any file in this repo without asking about that specific edit
first, and waiting for an answer.** Not "here is a list of 6 fixes, shall I?" —
each edit gets its own question, described concretely (what file, what line,
what changes), and the user approves or rejects it on its own.

This applies even when the user says "can you fix it." A general yes is NOT
approval for a batch of edits. It is permission to *propose* the first edit.
If the user asks for something that needs several changes, describe them and
ask which one to do first. Do not chain them.

This was violated on 2026-09-22 (a batch of edits across `transformer.py`,
`training.py`, `adapters.py`, and `training_loop.ipynb` shipped off one "can you
fix"). Do not repeat it. The whole point of this repo is that the user writes
the code; every edit taken away from them is a lost rep.

### No AI comments or sprawl in the user's files

- **Don't add comments to the user's code** unless they explain something
  genuinely non-obvious (e.g. why a regex alternative needs look-ahead). No
  narrating comments, no "this used to be X" history, no long docstrings.
- **Helpers and non-trivial code Claude writes go in a separate file** —
  `claude_suggested_speedups.py` — imported by the user's module, which keeps
  only one-line calls. The user's own functions (`merge_chunk`, `replace_pair`)
  are passed in so they stay the primitives. Don't let Claude's code swamp theirs.

- **Do not write assignment code.** Do not fill in `NotImplementedError` bodies,
  do not implement functions in `transformer.py` / `training.py` / `tokenizer.py`,
  and do not "fix" a failing test by writing the answer.
- **Do** explain intuition, point at the relevant section of the handout, ask
  Socratic questions, read code and describe what it does, diagnose *why* a test
  fails (without writing the fix), and help with plumbing that isn't the
  assignment (imports, env, wandb setup, plotting, file paths).
- **Ask when something looks unintuitive** rather than silently correcting it.
  The user explicitly asked for this. A weird-looking choice is often a
  deliberate one worth discussing.
- When pointing out a bug, describe the symptom and the reasoning that finds it;
  let the user write the patch.

## Course / assignment context

- **Course:** Stanford CS336, *Language Modeling from Scratch* (Spring 2025).
  Assignment 1 builds a complete LM stack with no `torch.nn` conveniences —
  no `nn.Linear`, no `nn.Embedding`, no `F.softmax`, no `torch.optim.AdamW`,
  no HuggingFace tokenizers. Only raw tensor ops, `einops`, and `nn.Module`/
  `nn.Parameter` scaffolding.
- **Handout:** `../cs336_spring2025_assignment1_basics.pdf` (in the repo root).
- **Grading interface:** `../tests/adapters.py`. Every problem is graded by
  wiring a thin adapter function to the user's implementation; the tests
  (`../tests/test_*.py`) compare against fixture snapshots in
  `../tests/_snapshots/` and `../tests/fixtures/`.
- **Data** lives in `../data/` (TinyStories ~2.2GB, OpenWebText ~12GB, plus
  trained BPE artifacts in `owt_bpe_32k/` and `tinystories_bpe_10k/`).

### Problem list (from the handout)

| § | Problems |
|---|---|
| 2 BPE Tokenizer | `unicode1`, `unicode2`, `train_bpe` (15pt), `train_bpe_tinystories`, `train_bpe_expts_owt`, `tokenizer` (15pt), `tokenizer_experiments` |
| 3 Architecture | `linear`, `embedding`, `rmsnorm`, `positionwise_feedforward`, `rope`, `softmax`, `scaled_dot_product_attention`, `multihead_self_attention` (5pt), `transformer_block`, `transformer_lm`, `transformer_accounting` (5pt) |
| 4 Training | `cross_entropy`, `learning_rate_tuning`, `adamw`, `adamwAccounting`, `learning_rate_schedule`, `gradient_clipping` |
| 5 Training loop | `data_loading`, `checkpointing`, `training_together` (4pt) |
| 6 Generation | `decoding` (3pt) |
| 7 Experiments | `experiment_log`, `learning_rate`, `batch_size_experiment`, `generate`, `layer_norm_ablation`, `pre_norm_ablation`, `no_pos_emb`, `swiglu_ablation`, `main_experiment`, `leaderboard` (6pt) |

## Repo layout

```
cs336_basics/
  tokenizer.py        BPE training (§2.4) + Tokenizer class (§2.6)
  claude_suggested_speedups.py
                      Claude-written speedups tokenizer.py calls into:
                      PretokenEncoder, stream_encode, run_bpe_merges
  transformer.py      All model modules (§3)
  training.py         Loss, AdamW, LR schedule, clipping, data loading,
                      checkpointing, decoding (§4–6) — the graded implementations
  train.py            TrainConfig dataclass + train() + argparse CLI. Holds the
                      training loop so a §7 sweep is a loop over configs.
                      Also exports ROOT / DATA path constants.
  training_loop.ipynb Data prep + inspection only: BPE training (guarded),
                      encode profiling, tokenizing to memmap, then calls train()
tests/adapters.py     Graded interface — thin wrappers over the above
  modal_app.py        Modal launcher, ~63 lines: image + Volume `cs336-a1` +
                      `tokenize_corpus` (CPU) + `train_run` (GPU, auto-resumes).
                      Launch = DEPLOY + SPAWN (notebook `launch(configs, gpu)`), never
                      `modal run --detach` + `.map()`: on 2026-09-24 the Mac slept, the
                      local clients died, and Modal cancelled both running jobs. Spawned
                      calls on the deployed app survive the laptop. Call IDs -> logs/*.json;
                      `results(calls)` or modal.FunctionCall.from_id(id).get().
```

## Status (as of 2026-09-22)

Test suite: **46 passed, 2 skipped, 0 failed** (`uv run pytest -q` from repo root).
Nothing in `train.py` or the notebook is graded; the adapters point only at
`tokenizer.py`, `transformer.py`, and `training.py`.

**Done and passing:** all of §2–§6 — BPE training (with multiprocessing
pretokenization), the Tokenizer, Linear, Embedding, RMSNorm, SwiGLU FFN, RoPE,
softmax, scaled dot-product attention, causal MHA (with and without RoPE), the
Transformer block, the full LM, cross-entropy, AdamW, cosine LR schedule,
gradient clipping, data loading, checkpointing, decoding.

**Not done:**
- All of §7. No completed training run, no wandb logs committed, no ablations.
- All written problems (`unicode1`, `unicode2`, `transformer_accounting`,
  `adamwAccounting`, `learning_rate_tuning`, `tokenizer_experiments`) — there is
  no writeup file in the repo yet.
- OWT tokenization never finished. The 2MB leftover from an interrupted run was
  renamed to `../data/owt_train_tokens.bin.partial` (the notebook's exists-guard
  would otherwise have trusted it); `tokenize_to_memmap` now writes `.tmp` and
  renames on completion. The encoder was the blocker; it is now fixed
  (see "Complexity fixes" below), and a full run is ~0.4h on one core.
  **Next: run it on Modal**, then train (§7.4).
- **Modal (2026-09-23):** `modal_app.py` built and verified end to end for
  tokenization. All four splits are on the Volume: ts_train 541,229,347 tokens (1.08GB), ts_valid 5,465,883, owt_train 2,727,120,451 (5.45GB), owt_valid 66,401,088. ts_valid.bin verified id-for-id against a local encode. Blockers for GPU training, both
  the user's to do: (1) **add a payment method** — Modal rejects `H100!`
  functions without one; (2) `uv run modal secret create wandb WANDB_API_KEY=...`.
  Sharded parallel tokenization verified identical to `encode()` locally.
  Older note: no Modal code existed in this repo or its git history before this.
- **B200 support:** the Modal image pins `torch~=2.8.0`. Probed on Modal: the default
  PyPI wheels for 2.6 (cu124) and 2.7 (cu126) lack sm_100; 2.8 and 2.9 are cu128 with
  sm_90 + sm_100. Full test suite passes on torch 2.8. Local env stays on 2.6.
  Every training entrypoint takes `--gpu` (default `H100!`); `--gpu B200` for Blackwell.
- **Measured throughput (smoke, 1000 steps, 2026-09-23):** H100 TinyStories bs32 eager
  247K tok/s (327.68M tokens ~22 min); B200 OWT bs128 eager 412K tok/s (~13 min; 45 min
  ~= 1.0B tokens). `--compile` measured slower (133K / 344K) but 1000 steps includes
  ~30s of compilation, so steady-state is untested. TS val loss 2.18 after 1000 steps.
- Image pins: torch~=2.8.0, numpy 2.3.2, einops 0.8.1, regex 2025.7.34, tqdm 4.67.1,
  wandb 0.21.2 (wandb 0.30 removed `wandb.util.generate_id`; train.py now uses uuid).
- `train.py` writes `<run>/summary.json` (steps, loop seconds, tokens/s, best val).
- Runs 2026-09-23: `ts-baseline` (H100, lr 1e-3, bs32, 327.68M tokens): val 1.3975 at
  step 39k (target 1.45 met). `owt-main` (B200, same config on OWT): best val 4.1433,
  19.6 min, 278K tok/s. Leaderboard run deliberately NOT launched yet.
- 7.2 sweeps launched 22:56 via notebook `launch()`: `lr-*` (7 LRs 1e-4..1e-1, 1/4 budget)
  and `bs-*` (bs 1..1024, 40.96M tokens each, lr 1e-3*sqrt(bs/32)). Logs in ../logs/.
- The GitHub handout is now Spring 2026 (v26.0.3): budgets in B200-hours; leaderboard =
  45 min on one B200, beat val 5.0. §7.1-7.2 content unchanged from 2025.
- `train.py` now writes `config.json` at start and auto-resumes from `latest.pt` when the
  run dir has one (so reusing a run name continues it, locally too).
- Local wandb API (notebook `runs_table` / `plot_runs`) needs `uv run wandb login` once. The script
  that produced `../data/owt_bpe_32k/` (container paths `/cache/inputs`, resume
  support, `merge_log.txt`, `metadata.json`) is lost or lives elsewhere. The
  OWT tokenizer does NOT need retraining; tokenizing the corpus and training the
  model are what's left.

## Audit fixes, 2026-09-22 (user chose "High + Medium, #1-#11")

Each verified by running it, not just by tests:
- #1 notebook `tokenize_to_memmap`: write `.tmp`, `os.replace` when done.
- #2 `Tokenizer.decode`: `b"".join` instead of O(n^2) `bytes +=`.
- #3 `load_checkpoint`: `map_location="cpu"` (GPU checkpoints load on the Mac).
- #4 `decoding`: `@torch.no_grad()` (decorator form restores grad mode between yields).
- #5 RoPE output cast back to input dtype; bf16 forward+backward now works.
- #6 MHA passes `token_positions.unsqueeze(-2)` so batched positions broadcast
  over heads (was silently wrong when batch == num_heads).
- #7 `Transformer_LM` sizes RoPE with `max(max_seq_len, context_length)`.
- #8 cross_entropy: **reverted, no benefit** — measured peak memory 864 -> 866MB;
  the peak is in backward, not the forward copies.
- #9 AdamW in-place `m`/`v`/`p` updates: 278 -> 95 ms/step on 100M params (speed
  only; peak memory unchanged because the allocator reuses temporaries).
- #10 notebook tokenizer uses `pretoken_cache_size=200_000`.
- #11 `train.py` writes `<ckpt>.state.pkl` (best_val, numpy RNG state, wandb id);
  interrupted-then-resumed run is bit-identical to an uninterrupted one.
- Not fixed on purpose: KV cache (not required), weight decay on RMSNorm gains
  (matches the handout's AdamW). Low-priority #12-#20 were not in scope.

## Complexity fixes to `tokenizer.py`, 2026-09-22 (batch-approved by the user)

The machinery now lives in `claude_suggested_speedups.py` (moved there at the
user's request): `PretokenEncoder` (encode), `stream_encode` (encode_iterable),
`run_bpe_merges` (merge loop). Names below like `build_pair_index` are now
private helpers in that file.

Every change was checked for **identical output** against the original code
(encode ids, and merges + vocab at vocab 2000 and 5000 on TinyStories valid and
260 on the full 2.2GB TinyStories train), plus the pytest suites and a fuzzer
comparing `encode_iterable` over random chunkings against `encode`.

| | before | after |
|---|---|---|
| encode, 50KB OWT | 60.3s | 0.023s |
| encode OWT train (real 100MB throughput, cache=200K) | ~3,996h (extrapolated) | ~0.4h one core |
| BPE merge loop, vocab 5000 | 29.7s | 0.19s |
| pretok 2.2GB TinyStories, 9 workers | 67s, 1,147MB/worker | 25s, 313MB/worker |
| full pytest suite | 52s | 3s |

1. **`encode`**: `merge_ranks` dict + per-pretoken "apply the lowest-ranked pair
   present" loop (O(L^2) per pretoken vs O(len(merges) x L)), plus a
   `_pretoken_cache` keyed by pretoken string. Removed `del merges_remaining[0]`
   (O(V) per call).
2. **`encode_iterable`**: batches lines (`_batched`), cuts after *trusted*
   special tokens (ending >= max_special_len before the buffer end, so the
   overlapping `<|endoftext|><|endoftext|>` case is safe), and on a long
   special-free stretch cuts at a pretoken boundary short of the tail.
   `PRETOKEN_REACH = 4` exists because the contraction alternative reads up to
   3 chars ahead. Two real bugs the fuzzer caught while building it: a negative
   `endpos` (the `regex` module counts it from the end, unlike stdlib `re`), and
   the contraction look-ahead splitting `'ve` into `'` + `ve`.
3. **Merge loop**: `build_pair_index` + `merge_pair_everywhere` (inverted index
   pair -> word ids, so a merge only visits words containing the pair), a heap
   with lazy deletion (push on *every* count change, including decreases), and
   `new_token_id = len(vocab)`. `_Descending` inverts bytes ordering so the
   min-heap reproduces the "lexicographically greatest" tiebreak exactly.
4. **Pretokenization**: `chunk_bytes` (default 32MB) decouples chunk count from
   worker count; `imap_unordered` merges results as they arrive; workers count
   pretoken *strings* with `Counter` and stopped computing pair counts (the merge
   loop builds them).
5. **Minor**: `Tokenizer.__init__` no longer sorts the caller's special_tokens
   list in place (and the `= []` mutable default is now `None`); removed
   `resolve_token`, whose recursive branch could never run.

**Knob:** `pretoken_cache_size` defaults to 4096 (~0.65MB) to fit the 1MB
budget of `test_encode_iterable_memory_usage`. Pass `200_000` for the OWT job:
3.5x faster on real data.

**Unverified:** `test_encode_iterable_memory_usage` and
`test_encode_memory_usage` are Linux-only (RLIMIT_AS) and skipped on macOS, so
they have not been run against the new code. Run them on Modal/Linux.

Backups and verification harness from that session lived in the session
scratchpad (not the repo): original `tokenizer.py`, `bench.py`,
`fuzz_iterable.py`. The original is also recoverable with
`git show HEAD:cs336_basics/tokenizer.py`.

## Refactor of 2026-09-22 (batch-approved by the user)

`training_loop.ipynb` was rewritten and `train.py` added, addressing 11 structural
points plus an OWT BPE training cell:

- Long jobs are **guarded** — `train_bpe_if_needed` and `tokenize_to_memmap` both
  early-return if their artifacts exist. "Run All" is now safe.
- New OWT 32K BPE training cell (§2.5 `train_bpe_expts_owt`), guarded the same way.
- `train_data.max()` on the memmap (which paged in the whole file) replaced by a
  1M-token sample check; the real validation now happens inside
  `tokenize_to_memmap`, where the ids are already in hand.
- Training loop extracted to `train.py` behind `TrainConfig`, so the §7 sweeps are
  `for lr in ...: train(dataclasses.replace(cfg, ...))`.
- `set_seed()` seeds numpy + torch; the seed is part of the config and is logged.
- Evaluation uses **fixed** batches (`make_eval_starts`, its own `default_rng` so
  it never perturbs the training stream), so val curves are comparable run to run.
  `estimate_loss` now toggles `model.eval()` / `model.train()`.
- All notebook imports consolidated into one cell; `ROOT`/`DATA` derived from
  `__file__`, so nothing depends on Jupyter's cwd.
- `wandb.finish()` moved into a `finally` so a crash still closes the run.
- `resume_from` restores model + optimizer + iteration and continues the loop.
- Checkpoints pruned to `latest.pt` + `best.pt` (`keep_only_latest_and_best`).
- Generate section scaffolded (loads `best.pt`, samples with temp/top-p).
  **The §7.2 writeup is still the user's to do.**

Verified: `train()` and the `python -m cs336_basics.train` CLI both run, resume
works (checkpoint at iter 6 -> resumed at 7), initial loss ~10.37 = ln(32000) as
expected for random init.

**Not gitignored yet: `checkpoints/` and `wandb/`.** Worth adding before any real
run — a single checkpoint is ~3x model size (Adam m and v ride along).

## Fixes applied 2026-09-22 (at the user's request)

These were plumbing/bug fixes on already-solved problems, not new solutions:

- `training.py`: imports were `from tokenizer import ...`, which only resolved
  when cwd was `cs336_basics/` and broke collection of the *entire* test suite.
  Now `from cs336_basics.X import ...`.
- `MultiHead_Self_Attention` gained `use_rope: bool = True`; `max_seq_len` and
  `theta` are now optional. This unblocked `run_multihead_self_attention` (the
  no-RoPE adapter) and is the hook the `no_pos_emb` ablation will need.
- The causal mask is now built with `device=x.device` (it was CPU-only, which
  would break on MPS/CUDA).
- `softmax` and `cross_entropy` no longer mutate their input in place.
- `run_silu` and `run_multihead_self_attention` wired in `tests/adapters.py`.
- `decoding`: used `model.d_model` where it meant `context_length` (now
  `model.context_length`, newly stored on `Transformer_LM`); dropped the
  `\x00` left-padding of short prompts; fixed `.view()` on a Python `int`;
  the window now only truncates at the model call instead of sliding every step;
  top-p uses an exclusive cumsum so the token crossing the threshold is kept.
- `training_loop.ipynb`: `import tqdm` -> `from tqdm.auto import tqdm`.

## Open questions for the user

3. `MultiHead_Self_Attention` passes `Linear(num_heads*d_k, d_model)`; those are
   equal so it works, but the arg order reads backwards vs `Linear(in, out)`.
4. `AdamW` stores `t` in `param_groups` rather than `self.state[p]`.
5. The notebook jumps straight to OWT (§7.4) with `d_model=128, d_ff=512`,
   skipping TinyStories (§7.2) and the handout's `d_model=512, d_ff=1344`.
   Deliberate "fit on MPS" choice, or placeholder?

## Conventions in this codebase

- `einops` (`einsum`/`rearrange`/`reduce`/`repeat`) is preferred over raw
  `torch` reshape/permute/matmul throughout.
- Parameter attributes use the user's own names (`weights`, `embedding_matrix`,
  `rmsnorm_gains`), not torch's `weight`/`bias` — `adapters.py` maps the
  reference state dicts onto these.
- Class names are non-PEP8 by choice: `RMSnorm`, `MultiHead_Self_Attention`,
  `Transformer_Block`, `Transformer_LM`.
- Google-style docstrings with tensor shapes on nearly every function; comments
  sometimes cite the reference the user learned from. That's the user's style for
  their own code — it is not license for Claude to add comments (see above).

## Commands

```bash
uv run pytest -q                      # full suite (~3s)
uv run pytest -q tests/test_model.py  # architecture only
uv run pytest -q -k rope              # single problem
./make_submission.sh                  # build the submission bundle
```

## Leaderboard / OWT trials (2026-09-23/24)

- `train.py` time mode: `time_budget_minutes` runs the cosine schedule on elapsed-time
  fraction (lr_schedule(frac, ..., warmup_frac, 1.0)); 20 evals; elapsed saved in
  state.pkl so resume keeps the clock. `autocast` (bf16) and `save_checkpoints` flags.
- Round 1 (8 min, B200, base model): best lr ~4e-3 at bs128; bs256 ~ties. autocast
  alone slower (389K vs 405K tok/s); compile 598K; autocast+compile 716K (1.77x) and
  best loss (3.957).
- Leaderboard run `lb-final-amp-compile` (bs128, lr 4e-3, amp+compile, 44 min budget):
  interrupted at 41.82 min (val 3.7507) by the laptop-sleep cancellation, resumed via
  spawn to finish its last ~2 min. `lb-bs256-lr0.004-38min` (plain) resumed at 13.37 min.
- TinyStories: at full budget lr 1e-3 (1.3957) beat 3e-3 (1.4126) although 3e-3 won at
  1/4 budget; best-LR shifts lower with longer runs.
