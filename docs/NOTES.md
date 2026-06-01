# GRPO, briefly

GRPO (Group Relative Policy Optimization, from DeepSeekMath and used in R1) is
PPO with the value network removed. The baseline that PPO normally learns with a
critic is replaced by the mean reward of a group of samples drawn from the same
prompt. That is the whole trick, and it is why this fits on a laptop: no critic
to train or store.

## The loop

For each prompt q:

1. Sample a group of G completions from the current policy.
2. Score each completion with a programmatic reward r_i (see below). No reward
   model, the reward is computed by a verifier.
3. Group-relative advantage, the same scalar for every token of completion i:

       A_i = (r_i - mean(r)) / (std(r) + eps)

   A group where every completion scores the same gives zero advantage, which is
   correct: there is nothing to prefer within it.
4. PPO clipped objective on per-token log-prob ratios against the sampling
   policy, minus a KL penalty against a frozen reference model:

       ratio = exp(logp_policy - logp_old)
       L_clip = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
       KL     = exp(d) - d - 1,   d = logp_ref - logp_policy      (k3 estimator)
       loss   = - mean_over_tokens(L_clip - beta * KL)

The KL term keeps the policy from drifting into degenerate text while it chases
reward. beta trades reasoning gains against staying close to the reference.

## Verifiable rewards

The reward here is two parts (see `rewards.py`):

- correctness: parse the number inside `<answer>...</answer>` and compare it to
  the gold answer. Exact, no judgment call.
- format: a small bonus for actually using `<think>...</think><answer>...</answer>`.

Because the reward is a function, not a learned model, the signal cannot be
gamed by fooling a reward model, and runs are reproducible. The cost is that it
only works on domains with a checkable answer. Math (GSM8K) is the cleanest
starting point. Code with unit tests is the natural next domain (see ROADMAP).

## References

- DeepSeekMath: https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- KL estimators (Schulman): http://joschu.net/blog/kl-approx.html
