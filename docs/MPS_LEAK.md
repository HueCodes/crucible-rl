# MPS memory growth: investigation and resolution

**Resolution:** the root cause was MPSGraph caching a compiled graph per input
shape for the embedding (and matmul) backward. It was not an allocator leak and
not an `empty_cache` bug. It is already fixed in the PyTorch nightly
(2.13.0.dev20260601), verified below. On the released 2.12.0 the workaround is the
memory watchdog plus `scripts/supervise.sh`. After upgrading to torch 2.13 stable,
the workaround can be removed. No upstream issue or PR is needed.

## What happened

fp32 GRPO training on MPS (M5 Pro, 48GB, torch 2.12.0) grew driver-allocated
memory about 1.5GB/step and OOM'd around step 28, despite `gc.collect()` and
`torch.mps.empty_cache()` every step. A fresh process reset it.

## What it was NOT (corrected from earlier guesses)

- Not our code. A standalone repro with zero Crucible code reproduces it
  (`scripts/mps_leak_repro.py`).
- Not a live-tensor leak. `current_allocated_memory()` stays flat (~4.4GB); only
  `driver_allocated_memory()` grows. The growth is reserved memory, not live tensors.
- Not an allocator/`empty_cache` bug. `empty_cache` cannot reclaim it because the
  memory is held by MPSGraph's compiled-graph cache, which the caching allocator
  does not own.

## Root cause

MPSGraph compiles and caches one graph per distinct input shape. RL rollouts have a
different completion length every step, so each new shape adds a cached graph and
its buffers. The embedding backward was the dominant offender: it cached a
vocab-by-dim gradient buffer per shape. Same class as pytorch/pytorch#181213
(closed, completed), where the team fixed gelu, softmax, and SDPA by adding Metal
kernel paths.

## Evidence

`scripts/mps_op_isolate.py`, varying only sequence length with weights fixed,
driver-memory growth over the loop:

| op | torch 2.12.0 | torch 2.13.0.dev |
|----|--------------|------------------|
| logsumexp | flat | flat |
| matmul | +91 MB (PREFER_METAL: flat) | +90 MB (PREFER_METAL: flat) |
| embedding backward | +1746 MB | flat (fixed) |
| full model (40 steps) | +18,500 MB | +15 MB (fixed) |

On nightly the full model held flat over 80 steps (5537 to 5571 MB). On 2.12.0 the
same loop OOMs. `PYTORCH_MPS_PREFER_METAL=1` covers matmul but not embedding on
2.12.0, so it is not a full fix there. The residual matmul growth is bounded (the
full model with a matmul head stays flat), so it is not a concern.

## What to do

- On 2.12.0 (current): keep the watchdog (`Config.mem_restart_gb`) plus
  `scripts/supervise.sh`. Verified to complete a 400-step run.
- On torch 2.13 stable (when released): upgrade, then remove `mem_restart_gb` and
  the supervisor. The leak is gone.

## Environment

Apple M5 Pro, 48GB, macOS 26.4. Affected: torch 2.12.0. Verified fixed:
torch 2.13.0.dev20260601. Diagnostics: `scripts/mps_leak_repro.py`,
`scripts/mps_op_isolate.py`.
