from __future__ import annotations

import torch

from .utils import masked_mean


def sequence_logprobs(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Per-token log prob of each realized token under `model`.

    Returns a [B, T-1] tensor aligned so that position t holds log p(token_{t+1}).
    Pair it with a shifted completion mask to score only generated tokens.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    # log p(target) = logit[target] - logsumexp(logits). Computing it this way
    # avoids materializing the full [B, T, vocab] log_softmax tensor, at this
    # model's ~152k vocab that intermediate is ~1GB per call and was OOMing MPS.
    tgt_logit = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return tgt_logit - torch.logsumexp(logits, dim=-1)


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
    # The estimator is differentiated as part of the objective, so a single token
    # where the policy assigns far lower probability than the reference (d >> 0)
    # makes exp(d), and its gradient, blow up, which after grad-norm clipping
    # crushes the real policy signal and sends training divergent. Clamp the
    # exponent to a sane band so outlier tokens can't dominate the step.
    d = (ref_logp - policy_logp).clamp(-10.0, 10.0)
    kl = torch.exp(d) - d - 1.0

    per_token = policy_term - beta_kl * kl
    loss = -masked_mean(per_token, mask)
    metrics = {
        "loss": float(loss.detach()),
        "kl": float(masked_mean(kl, mask).detach()),
        "ratio": float(masked_mean(ratio, mask).detach()),
    }
    return loss, metrics
