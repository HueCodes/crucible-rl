from __future__ import annotations

import argparse
import os
import random

# MPS lacks a few ops used by some models; fall back to CPU for those instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from . import config as config_mod
from .data import load_gsm8k
from .eval import accuracy
from .grpo import grpo_loss, group_advantages, sequence_logprobs
from .metrics import JsonlLogger
from .model import encode_prompt, generate_group, load_policy_and_ref
from .rewards import score
from .utils import pick_device, set_seed


def _parse_args():
    p = argparse.ArgumentParser(description="GRPO reasoning trainer")
    p.add_argument("--preset", default="smoke", choices=sorted(config_mod.PRESETS))
    p.add_argument("--model", dest="model_name")
    p.add_argument("--steps", dest="total_steps", type=int)
    p.add_argument("--group-size", dest="group_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "preset" and v is not None}
    return config_mod.load(args.preset, **overrides)


def rollout_and_loss(cfg, tok, policy, ref, sample, device):
    """One prompt: sample a group, score it, return the GRPO loss and stats."""
    prompt_ids = encode_prompt(tok, sample.question, cfg, device)
    out, attn, comp_mask, texts = generate_group(policy, tok, prompt_ids, cfg)

    breakdowns = [score(t, sample.gold, cfg.correct_reward, cfg.format_reward) for t in texts]
    rewards = torch.tensor([b.total for b in breakdowns], device=device, dtype=torch.float32)
    advantages = group_advantages(rewards)

    shift_mask = comp_mask[:, 1:].float()
    with torch.no_grad():
        old_logp = sequence_logprobs(policy, out, attn)
        ref_logp = sequence_logprobs(ref, out, attn)
    policy_logp = sequence_logprobs(policy, out, attn)
    loss, metrics = grpo_loss(
        policy_logp, old_logp, ref_logp, advantages, shift_mask, cfg.clip_eps, cfg.beta_kl
    )
    metrics["reward"] = float(rewards.mean())
    metrics["acc"] = float((torch.tensor([b.correct for b in breakdowns]) > 0).float().mean())
    return loss, metrics


def main():
    cfg = _parse_args()
    set_seed(cfg.seed)
    device = pick_device()
    os.makedirs(cfg.save_dir, exist_ok=True)
    logger = JsonlLogger(cfg.save_dir)
    print(f"device={device} preset model={cfg.model_name}")

    tok, policy, ref = load_policy_and_ref(cfg, device)
    train_set = load_gsm8k(cfg.dataset_name, cfg.dataset_config, "train")
    eval_set = load_gsm8k(cfg.dataset_name, cfg.dataset_config, "test")[: cfg.eval_size]

    opt = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=cfg.lr
    )

    base_acc = accuracy(policy, tok, eval_set, cfg, device)
    print(f"step=0 eval_acc={base_acc:.3f} (baseline)")
    logger.log(0, "eval", acc=base_acc)

    for step in range(1, cfg.total_steps + 1):
        policy.train()
        opt.zero_grad()
        batch = random.sample(train_set, cfg.prompts_per_step)
        agg = {"loss": 0.0, "kl": 0.0, "reward": 0.0, "acc": 0.0}
        for sample in batch:
            for _ in range(cfg.inner_epochs):
                loss, m = rollout_and_loss(cfg, tok, policy, ref, sample, device)
                (loss / (cfg.prompts_per_step * cfg.inner_epochs)).backward()
            for k in agg:
                agg[k] += m[k] / cfg.prompts_per_step

        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
        opt.step()

        logger.log(step, "train", loss=agg["loss"], reward=agg["reward"], acc=agg["acc"], kl=agg["kl"])
        if step % cfg.log_every == 0:
            print(
                f"step={step} loss={agg['loss']:.4f} reward={agg['reward']:.3f} "
                f"acc={agg['acc']:.3f} kl={agg['kl']:.4f}"
            )
        if step % cfg.eval_every == 0:
            acc = accuracy(policy, tok, eval_set, cfg, device)
            print(f"step={step} eval_acc={acc:.3f}")
            logger.log(step, "eval", acc=acc)

    policy.save_pretrained(cfg.save_dir)
    tok.save_pretrained(cfg.save_dir)
    print(f"saved policy to {cfg.save_dir}")


if __name__ == "__main__":
    main()
