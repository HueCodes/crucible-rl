from __future__ import annotations

import json
import os
import random

import numpy as np
import torch

# Resumable checkpoint: model + optimizer + RNG + step, in one file we overwrite
# each save. Separate from the final save_pretrained() (which writes a clean HF
# model for inference); this one exists so a multi-hour run survives a crash.
CKPT_NAME = "checkpoint.pt"
STATE_NAME = "trainer_state.json"


def save_checkpoint(save_dir: str, policy, opt, step: int) -> None:
    """Atomically write a resumable checkpoint (write-temp-then-rename so a crash
    mid-write can't corrupt the previous good checkpoint)."""
    payload = {
        "model": policy.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
    }
    tmp = os.path.join(save_dir, CKPT_NAME + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, os.path.join(save_dir, CKPT_NAME))
    with open(os.path.join(save_dir, STATE_NAME), "w") as fh:
        json.dump({"step": step}, fh)


def load_checkpoint(save_dir: str, policy, opt, device: str) -> int:
    """Restore from save_dir/checkpoint.pt if present. Returns the last completed
    step (0 if there is nothing to resume), so the loop continues from step+1."""
    path = os.path.join(save_dir, CKPT_NAME)
    if not os.path.exists(path):
        return 0
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    # Optimizer state tensors load on CPU; move them to the training device so the
    # first post-resume step doesn't hit a device mismatch.
    for state in opt.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    random.setstate(ckpt["rng_python"])
    np.random.set_state(ckpt["rng_numpy"])
    torch.set_rng_state(ckpt["rng_torch"])
    return int(ckpt["step"])
