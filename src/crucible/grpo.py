from __future__ import annotations

import torch
import torch.nn.functional as F

from .utils import masked_mean


def sequence_logprobs(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Per-token log prob of each realized token under `model`.

    Returns a [B, T-1] tensor aligned so that position t holds log p(token_{t+1}).
    Pair it with a shifted completion mask to score only generated tokens.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """GRPO advantage: standardize rewards within the group, no value network.

    rewards: [G]. Returns [G]. A constant-reward group yields zero advantage,
    which is correct (nothing to prefer within it).
    """
    return (rewards - rewards.mean()) / (rewards.std() + 1e-4)


def grpo_loss(
    policy_logp: torch.Tensor,   # [B, T-1] with grad
    old_logp: torch.Tensor,      # [B, T-1] detached, from sampling policy
    ref_logp: torch.Tensor,      # [B, T-1] detached, from frozen reference
    advantages: torch.Tensor,    # [B] one scalar advantage per sequence
    mask: torch.Tensor,          # [B, T-1] 1 on completion tokens
    clip_eps: float,
    beta_kl: float,
):
    ratio = torch.exp(policy_logp - old_logp)
    adv = advantages.unsqueeze(1)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_term = torch.minimum(unclipped, clipped)

    # k3 unbiased KL estimator (Schulman): exp(d) - d - 1, d = logp_ref - logp_policy.
    d = ref_logp - policy_logp
    kl = torch.exp(d) - d - 1.0

    per_token = policy_term - beta_kl * kl
    loss = -masked_mean(per_token, mask)
    metrics = {
        "loss": float(loss.detach()),
        "kl": float(masked_mean(kl, mask).detach()),
        "ratio": float(masked_mean(ratio, mask).detach()),
    }
    return loss, metrics
