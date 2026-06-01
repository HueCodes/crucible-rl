"""Fast checks for the verifier and the GRPO math. No model download required.

Run with:  uv run pytest        (or)   uv run python tests/test_rewards.py
"""
from __future__ import annotations

import torch

from crucible.grpo import grpo_loss, group_advantages
from crucible.rewards import correctness_reward, extract_answer, format_reward, score

GOOD = "<think>2 plus 2 is 4</think><answer>4</answer>"
WRONG = "<think>guessing</think><answer>5</answer>"
UNFORMATTED = "the answer is 4"


def test_extract_answer():
    assert extract_answer(GOOD) == "4"
    assert extract_answer("<answer>1,024</answer>") == "1024"
    assert extract_answer("<answer>3.5</answer>") == "3.5"
    assert extract_answer("<answer>nope</answer>") is None
    # falls back to scanning raw text when no tags are present
    assert extract_answer(UNFORMATTED) == "4"


def test_correctness_reward():
    assert correctness_reward(GOOD, "4", 1.0) == 1.0
    assert correctness_reward(WRONG, "4", 1.0) == 0.0
    # numeric equality, not string equality
    assert correctness_reward("<answer>4.0</answer>", "4", 1.0) == 1.0


def test_format_reward():
    assert format_reward(GOOD, 0.5) == 0.5
    assert format_reward(UNFORMATTED, 0.5) == 0.0
    # answer without preceding think block does not satisfy the format
    assert format_reward("<answer>4</answer>", 0.5) == 0.0


def test_score_combines():
    b = score(GOOD, "4", correct_w=1.0, format_w=0.5)
    assert b.total == 1.5 and b.correct == 1.0 and b.format == 0.5


def test_group_advantages_zero_mean_unit_scale():
    adv = group_advantages(torch.tensor([0.0, 1.0, 0.0, 1.0]))
    assert abs(float(adv.mean())) < 1e-5
    assert float(adv.std()) > 0.0


def test_group_advantages_constant_group_is_flat():
    adv = group_advantages(torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-3)


def test_grpo_loss_shapes_and_kl_zero_at_ref():
    b, t = 4, 6
    logp = torch.randn(b, t)
    mask = torch.ones(b, t)
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    # policy == old == ref: ratio is 1 and KL is 0, so loss is -mean(adv*1) over tokens
    loss, m = grpo_loss(logp, logp, logp, adv, mask, clip_eps=0.2, beta_kl=0.04)
    assert abs(m["kl"]) < 1e-5
    assert abs(m["ratio"] - 1.0) < 1e-5
    assert torch.isfinite(loss)


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")
    sys.exit(0)
