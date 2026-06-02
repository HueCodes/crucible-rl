"""Isolate which ops grow the MPS MPSGraph cache under varying input shapes.

For each op, run a forward+backward loop where only the sequence length varies
(weights fixed outside the loop), calling empty_cache() each step, and report the
growth in torch.mps.driver_allocated_memory(). Ops that route through MPSGraph
cache a compiled graph per shape and grow unbounded; ops with a Metal kernel path
stay flat.

Run the same script under different torch builds to compare (e.g. released vs
nightly), and with PYTORCH_MPS_PREFER_METAL=1 to see which ops the flag covers:

    .venv/bin/python scripts/mps_op_isolate.py
    PYTORCH_MPS_PREFER_METAL=1 /tmp/torch-nightly/bin/python scripts/mps_op_isolate.py
"""
from __future__ import annotations

import os
import random

import torch
import torch.nn as nn

DEV = "mps"
G, V, D = 4, 8192, 2048
STEPS = 30


def drv_mb() -> float:
    return torch.mps.driver_allocated_memory() / 1e6


def isolate(name, setup, body, steps=STEPS):
    random.seed(0)
    torch.manual_seed(0)
    torch.mps.empty_cache()
    ctx = setup()
    base = end = 0.0
    for s in range(1, steps + 1):
        body(ctx, random.randint(64, 256))
        torch.mps.empty_cache()
        if s == 3:
            base = drv_mb()
        if s == steps:
            end = drv_mb()
    flag = "GROWS" if end - base > 20 else "flat"
    print(f"  {name:20s} {end - base:+9.1f} MB   [{flag}]")


class Tiny(nn.Module):
    def __init__(self, vocab=151936, hid=896):
        super().__init__()
        self.emb = nn.Embedding(vocab, hid)
        self.head = nn.Linear(hid, vocab)

    def forward(self, ids):
        return self.head(self.emb(ids))


def main():
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS not available")
    print(f"torch {torch.__version__}  PREFER_METAL={os.environ.get('PYTORCH_MPS_PREFER_METAL','0')}\n")

    isolate("logsumexp", lambda: None,
            lambda c, L: torch.logsumexp(
                torch.randn(G, L, V, device=DEV, requires_grad=True), -1).sum().backward())
    isolate("matmul (fixed w)", lambda: torch.randn(D, V, device=DEV),
            lambda w, L: (torch.randn(G, L, D, device=DEV, requires_grad=True) @ w).sum().backward())
    isolate("embedding (fixed)", lambda: nn.Embedding(V, D).to(DEV),
            lambda e, L: e(torch.randint(0, V, (G, L), device=DEV)).sum().backward())

    # full large-vocab model (embedding + linear head), the real training shape
    random.seed(0)
    torch.manual_seed(0)
    torch.mps.empty_cache()
    m = Tiny().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-6)
    base = end = 0.0
    for s in range(1, 41):
        ids = torch.randint(0, 151936, (4, random.randint(64, 256)), device=DEV)
        lg = m(ids)
        lp = (lg.gather(-1, ids.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(lg, -1)).mean()
        opt.zero_grad(set_to_none=True)
        (-lp).backward()
        opt.step()
        torch.mps.empty_cache()
        if s == 3:
            base = drv_mb()
        if s == 40:
            end = drv_mb()
    flag = "GROWS" if end - base > 100 else "flat"
    print(f"  {'full model':20s} {end - base:+9.1f} MB   [{flag}]")


if __name__ == "__main__":
    main()
