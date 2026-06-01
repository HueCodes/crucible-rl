"""Minimal, self-contained reproducer for the MPS memory growth we hit in training.

This deliberately uses NO Crucible code. It runs forward+backward+optimizer on a
tiny model on the MPS device, calling gc.collect()+empty_cache() every step (exactly
what the trainer does), and prints driver-allocated memory. It runs the loop twice:

  A) FIXED sequence length every step
  B) VARIABLE (random) sequence length every step  <- matches RL rollouts

Interpretation (this is the whole point — it isolates the cause):
  * B grows, A flat   -> the MPS caching allocator hoards a size-bucket per distinct
                         shape and empty_cache() doesn't reclaim it. Arguably intended
                         caching, but empty_cache() not releasing is a real complaint.
                         Workaround for us: pad rollouts to a fixed length.
  * A and B both grow -> a more general MPS allocation leak; strongest upstream bug case.
  * Neither grows     -> PyTorch is fine; our trainer (the rollouts list / refs) was
                         the leak. Fix it in this repo, no upstream work needed.

Run it ONLY when nothing else is using the GPU (it competes for the 48GB):
    .venv/bin/python scripts/mps_leak_repro.py

Record the two columns of mem= numbers; they are the evidence for the issue report.
"""
from __future__ import annotations

import gc
import random

import torch
import torch.nn as nn

# Match Qwen2.5-0.5B's large vocab — the [group, seq, vocab] logits are the big
# tensors and the most likely thing the allocator is bucketing by shape.
VOCAB = 151936
HIDDEN = 896
GROUP = 4
STEPS = 50


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, HIDDEN)
        self.head = nn.Linear(HIDDEN, VOCAB)

    def forward(self, ids):
        return self.head(self.emb(ids))


def run(label: str, fixed_len: int | None):
    torch.manual_seed(0)
    random.seed(0)
    model = Tiny().to("mps")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
    first = last = 0.0
    for s in range(1, STEPS + 1):
        length = fixed_len if fixed_len else random.randint(64, 256)
        ids = torch.randint(0, VOCAB, (GROUP, length), device="mps")
        logits = model(ids)
        # logit - logsumexp, same memory-efficient log-prob the trainer uses
        logp = (logits.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
                - torch.logsumexp(logits, dim=-1)).mean()
        opt.zero_grad(set_to_none=True)
        (-logp).backward()
        opt.step()
        gc.collect()
        torch.mps.empty_cache()
        mem = torch.mps.driver_allocated_memory() / 1024**3
        if s == 1:
            first = mem
        last = mem
        if s == 1 or s % 5 == 0:
            print(f"[{label:8s}] step {s:3d}  len={length:3d}  mem={mem:5.1f} GB")
    print(f"[{label:8s}] growth over {STEPS} steps: {last - first:+.1f} GB "
          f"({first:.1f} -> {last:.1f})\n")
    del model, opt
    gc.collect()
    torch.mps.empty_cache()


def main():
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS not available — run this on the Apple Silicon machine.")
    print(f"torch {torch.__version__}\n")
    run("fixed", fixed_len=256)
    run("variable", fixed_len=None)


if __name__ == "__main__":
    main()
