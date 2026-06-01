# MPS memory growth — investigation plan

**Status:** parked until the GSM8K run finishes (don't compete for the 48GB GPU).
Currently *worked around* by a memory watchdog + auto-resume supervisor; this doc is
about finding the *root cause* and possibly contributing a fix upstream.

## What we observed

During fp32 GRPO training on MPS (M5 Pro, 48GB, torch 2.12.0, macOS 26.4), driver
memory climbs ~1.5–2.5 GB per step and is **not** reclaimed by `gc.collect()` +
`torch.mps.empty_cache()` (both called every step). It OOMs around step ~28 from a
fresh process. A fresh process resets it (e.g. 42 GB → 24 GB across a restart).

Measured trend (one run, group 4 / 256 tok):

```
step 1  18.0 GB
step 6  30.2 GB
step 11 38.8 GB
step 16 46.5 GB
step 21 50.6 GB
step 26 55.0 GB   -> OOM at step 28
```

## Current workaround (in tree, working)

- `Config.mem_restart_gb` (42.0 for the gsm8k preset): when driver memory crosses it,
  the trainer checkpoints and exits with `RESTART_EXIT_CODE` (42).
- `scripts/supervise.sh` re-launches with `--resume`; a fresh process resets the leak.
- The run completes at full quality, spread across ~30 short-lived processes.

This is fine for getting runs done, but it costs ~30 process reloads and is a band-aid.

## Hypotheses (what we don't yet know)

1. **Variable-shape caching.** Each step's rollouts have different sequence lengths →
   new `[group, seq, vocab]` tensor shapes → the MPS caching allocator reserves a new
   size-bucket per shape and `empty_cache()` doesn't release them. Arguably "intended
   caching," but `empty_cache()` failing to reclaim is still a legitimate report.
2. **General MPS allocation leak.** A buffer that's never freed regardless of shape.
   This is the strongest upstream-bug case.
3. **Our code.** The `rollouts` list / cached tensors in `train.py` hold references
   across steps. If so it's *our* bug — fix it here, no upstream work.

## Step 1 — run the reproducer (decides everything)

`scripts/mps_leak_repro.py` is standalone (no Crucible code). It runs forward+backward
on a tiny large-vocab model with **fixed** vs **variable** sequence length and prints
the memory growth for each.

```
.venv/bin/python scripts/mps_leak_repro.py     # only when the GPU is otherwise idle
```

Decision tree from the two growth columns:

| fixed | variable | conclusion | next action |
|-------|----------|------------|-------------|
| flat  | grows    | shape-bucket caching / `empty_cache` not reclaiming (H1) | upstream report; **also** try padding our rollouts to fixed length (may remove the supervisor entirely) |
| grows | grows    | general MPS leak (H2) | strongest upstream report; consider source build + allocator fix |
| flat  | flat     | not PyTorch — it's our `rollouts` refs (H3) | fix in this repo; drop the supervisor |

## Step 2 — if it's PyTorch, contribute upstream

1. **Search first.** github.com/pytorch/pytorch issues for "MPS memory leak",
   "empty_cache mps not releasing", "mps memory grows". MPS memory reports are common —
   adding a *clean minimal repro* to an existing issue is high value on its own.
2. **File / comment** with: the repro output, `torch.__version__`, macOS version, chip,
   and the key fact that `empty_cache()` does not reclaim.
3. **Possible workaround to report alongside:** if H1, fixed-shape tensors avoid it.

## Step 3 — only if a code fix looks tractable: build from source

Feasible on this 48GB M5 Pro (Xcode + cmake + ninja already installed, 700GB free):

| Need | This machine |
|------|--------------|
| Disk | build ~15–25 GB; have ~700 GB |
| Build RAM | the one constraint — cap `MAX_JOBS=8`, 48GB is comfortable |
| First build | ~45–90 min (Mac build is MPS+CPU only, no CUDA) |
| Iterate | edit one file → incremental rebuild ~1–5 min |
| Test MPS code | **requires** Apple Silicon + Metal GPU — this machine is necessary, not just sufficient |

```
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch && pip install -r requirements.txt
MAX_JOBS=8 USE_CUDA=0 python setup.py develop
```

Allocator code to read: `aten/src/ATen/mps/MPSAllocator.mm` and `MPSAllocator.h`
(the `MPSHeapAllocatorImpl` — heap/buffer pooling, `EmptyCache`, the high/low
watermark logic). The likely-interesting path is what `EmptyCache` does and whether
buffers split into size-pools are ever released.

Do builds/tests only when the training run is **not** using the GPU.

## Environment

- Apple M5 Pro, 48 GB unified memory, macOS 26.4
- torch 2.12.0, MPS backend
- Repro: `scripts/mps_leak_repro.py`; workaround: `Config.mem_restart_gb` + `scripts/supervise.sh`
