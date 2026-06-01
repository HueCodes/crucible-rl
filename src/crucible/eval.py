from __future__ import annotations

import torch

from .config import Config
from .data import Sample
from .model import encode_prompt, generate_group
from .rewards import correctness_reward


@torch.no_grad()
def accuracy(policy, tok, samples: list[Sample], cfg: Config, device: str) -> float:
    """Greedy exact-match accuracy on a slice of held-out problems."""
    was_training = policy.training
    policy.eval()
    hits = 0
    for s in samples:
        prompt_ids = encode_prompt(tok, s.question, cfg, device)
        out, _, _, text = generate_group(policy, tok, prompt_ids, cfg, greedy=True)
        if correctness_reward(text[0], s.gold, weight=1.0) > 0:
            hits += 1
    if was_training:
        policy.train()
    return hits / max(len(samples), 1)
