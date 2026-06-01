from __future__ import annotations

import argparse
import gc
import os
import random
import sys

# Exit code the trainer uses to ask the supervisor for a fresh process (PyTorch's
# MPS allocator leaks across steps and empty_cache won't reclaim it; a restart is
# the only thing that resets it). Distinct from 0 (run complete) and other crashes.
RESTART_EXIT_CODE = 42

# MPS lacks a few ops used by some models; fall back to CPU for those instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from . import config as config_mod
from .checkpoint import load_checkpoint, save_checkpoint
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
    p.add_argument("--max-new-tokens", dest="max_new_tokens", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--resume", action="store_true", help="continue from save_dir/checkpoint.pt if present")
    args = p.parse_args()
    skip = {"preset", "resume"}
    overrides = {k: v for k, v in vars(args).items() if k not in skip and v is not None}
    if args.resume:
        overrides["resume"] = True
    return config_mod.load(args.preset, **overrides)


def _free_cache(device: str) -> None:
    """Release the allocator's cached blocks so each step's generation peak doesn't
    stack on the last and trip the MPS memory ceiling. gc.collect() first: dropped
    tensors from the previous step often aren't collected yet when empty_cache runs,
    so without it their blocks never return and 'other allocations' creep up to OOM."""
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def _mem_gb(device: str) -> float:
    """Driver-allocated memory in GiB, for watching the trend across steps."""
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 1024**3
    if device == "cuda":
        return torch.cuda.memory_reserved() / 1024**3
    return 0.0


def make_rollout(cfg, tok, policy, ref, sample, device):
    """Sample one prompt's group once and cache everything the inner-epoch updates
    reuse: tokens, group-relative advantages, the frozen reference's log-probs, and
    reward stats. `old_logp` (the behavior policy's log-probs) is filled in lazily
    on the first update from the policy itself, so the first epoch's ratio is
    exactly 1 and the PPO clip only bites once the weights have actually moved."""
    prompt_ids = encode_prompt(tok, sample.question, cfg, device)
    out, attn, comp_mask, texts = generate_group(policy, tok, prompt_ids, cfg)

    breakdowns = [score(t, sample.gold, cfg.correct_reward, cfg.format_reward) for t in texts]
    rewards = torch.tensor([b.total for b in breakdowns], device=device, dtype=torch.float32)
    with torch.no_grad():
        ref_logp = sequence_logprobs(ref, out, attn)
    return {
        "out": out,
        "attn": attn,
        "shift_mask": comp_mask[:, 1:].float(),
        "advantages": group_advantages(rewards),
        "ref_logp": ref_logp,
        "old_logp": None,
        "reward": float(rewards.mean()),
        "acc": float((torch.tensor([b.correct for b in breakdowns]) > 0).float().mean()),
    }


def rollout_loss(cfg, policy, r):
    """GRPO loss for a cached rollout under the *current* policy weights."""
    policy_logp = sequence_logprobs(policy, r["out"], r["attn"])
    if r["old_logp"] is None:
        r["old_logp"] = policy_logp.detach()
    loss, metrics = grpo_loss(
        policy_logp, r["old_logp"], r["ref_logp"], r["advantages"],
        r["shift_mask"], cfg.clip_eps, cfg.beta_kl,
    )
    metrics["reward"] = r["reward"]
    metrics["acc"] = r["acc"]
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

    start_step = load_checkpoint(cfg.save_dir, policy, opt, device) if cfg.resume else 0
    if start_step:
        print(f"resumed from checkpoint at step {start_step}")
    else:
        base_acc = accuracy(policy, tok, eval_set, cfg, device)
        print(f"step=0 eval_acc={base_acc:.3f} (baseline)")
        logger.log(0, "eval", acc=base_acc)
        _free_cache(device)

    for step in range(start_step + 1, cfg.total_steps + 1):
        policy.train()
        batch = random.sample(train_set, cfg.prompts_per_step)
        # Sample every prompt's group once, then reuse the rollouts across the
        # inner PPO epochs (each epoch is its own optimizer step on the same data).
        rollouts = [make_rollout(cfg, tok, policy, ref, s, device) for s in batch]

        agg = {"loss": 0.0, "kl": 0.0, "reward": 0.0, "acc": 0.0}
        any_skipped = False
        last_gnorm = 0.0
        for _epoch in range(cfg.inner_epochs):
            opt.zero_grad(set_to_none=True)
            ep = {"loss": 0.0, "kl": 0.0, "reward": 0.0, "acc": 0.0}
            for r in rollouts:
                loss, m = rollout_loss(cfg, policy, r)
                (loss / cfg.prompts_per_step).backward()
                for k in ep:
                    ep[k] += m[k] / cfg.prompts_per_step

            gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            # Don't let a single diverged update corrupt the policy: skip it if the
            # gradient is non-finite, pathologically large (a spike that signals an
            # unstable batch even though clipping bounds the applied step), or the
            # KL has blown past the guard. This is what keeps an unattended
            # overnight run from silently going off the rails.
            skipped = (
                (not torch.isfinite(gnorm))
                or (float(gnorm) > cfg.grad_norm_guard)
                or (ep["kl"] > cfg.kl_guard)
            )
            if skipped:
                opt.zero_grad(set_to_none=True)
                any_skipped = True
                print(f"step={step} SKIPPED (kl={ep['kl']:.2f} grad_norm={float(gnorm):.2f})")
            else:
                opt.step()
            last_gnorm = float(gnorm)
            for k in agg:
                agg[k] += ep[k] / cfg.inner_epochs

        logger.log(
            step, "train", loss=agg["loss"], reward=agg["reward"],
            acc=agg["acc"], kl=agg["kl"], grad_norm=last_gnorm, skipped=any_skipped,
        )
        if step % cfg.log_every == 0:
            print(
                f"step={step} loss={agg['loss']:.4f} reward={agg['reward']:.3f} "
                f"acc={agg['acc']:.3f} kl={agg['kl']:.4f}"
            )
        # Return cached allocator blocks every step: on unified-memory MPS they
        # otherwise accumulate and fragment across steps until an OOM (seen at
        # step 28 with ~44GiB of stale "other allocations" on a 48GB machine).
        _free_cache(device)
        mem_gb = _mem_gb(device)
        logger.log(step, "mem", mem_gb=round(mem_gb, 2))

        if step % cfg.eval_every == 0:
            acc = accuracy(policy, tok, eval_set, cfg, device)
            print(f"step={step} eval_acc={acc:.3f}")
            logger.log(step, "eval", acc=acc)
            save_checkpoint(cfg.save_dir, policy, opt, step)
            print(f"  checkpoint saved at step {step}")
            _free_cache(device)

        # Memory watchdog: the MPS allocator leaks ~1.5GB/step and won't release it,
        # so rather than crash into an OOM we checkpoint and ask the supervisor for a
        # fresh process the moment we approach the ceiling. A fresh process resets the
        # leak; --resume picks up exactly here. No steps are lost.
        if cfg.mem_restart_gb and mem_gb > cfg.mem_restart_gb and step < cfg.total_steps:
            save_checkpoint(cfg.save_dir, policy, opt, step)
            print(f"step={step} mem={mem_gb:.1f}GB > {cfg.mem_restart_gb}GB — "
                  f"checkpointed, restarting for a fresh allocator")
            sys.exit(RESTART_EXIT_CODE)

    policy.save_pretrained(cfg.save_dir)
    tok.save_pretrained(cfg.save_dir)
    print(f"saved policy to {cfg.save_dir}")


if __name__ == "__main__":
    main()
