from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass
class Config:
    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "bfloat16"  # bfloat16 | float16 | float32
    use_lora: bool = False

    # GRPO
    group_size: int = 8          # completions sampled per prompt (the "group")
    prompts_per_step: int = 4    # prompts per optimizer step (gradient accumulated over the groups)
    inner_epochs: int = 1        # policy updates per rollout batch (PPO-style reuse)
    clip_eps: float = 0.2        # PPO ratio clip
    beta_kl: float = 0.04        # KL penalty weight against the frozen reference
    kl_guard: float = 40.0       # skip an optimizer step whose mean KL exceeds this (divergence guard)
    grad_norm_guard: float = 1e3 # skip a step whose pre-clip grad norm spikes past this (instability guard)

    # Sampling
    temperature: float = 1.0
    top_p: float = 1.0
    max_prompt_len: int = 512
    max_new_tokens: int = 512

    # Optim
    lr: float = 1e-6
    max_grad_norm: float = 1.0
    total_steps: int = 500
    seed: int = 0
    resume: bool = False  # continue from save_dir/checkpoint.pt instead of starting fresh

    # Reward shaping
    correct_reward: float = 1.0
    format_reward: float = 0.5

    # Eval / logging / io
    eval_every: int = 25
    eval_size: int = 64
    log_every: int = 1
    save_dir: str = "runs/default"

    # Data
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"


# A tiny configuration that wires the whole loop end to end on a 0.5B model in a
# few minutes, so the first thing to run after setup proves nothing is broken.
SMOKE = Config(
    dtype="float32",  # match the real preset; bf16 RL is unstable on MPS (see GSM8K_0P5B)
    group_size=4,
    prompts_per_step=2,
    max_new_tokens=128,
    total_steps=2,
    eval_every=2,
    eval_size=8,
    save_dir="runs/smoke",
)

# The real starter run: 0.5B Qwen on GSM8K. Sized to leave headroom on a 64GB
# machine — group_size*max_new_tokens drives peak activation memory for backward,
# and the earlier 8x512 setting OOM'd MPS at ~61/64GiB. top_p<1 trims the
# degenerate-token tail that spikes the KL term.
GSM8K_0P5B = Config(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    dtype="float32",  # pure-bf16 RL overflowed logits/grads into NaN and poisoned the weights; fp32 is stable and cheap at 0.5B
    group_size=6,
    prompts_per_step=4,
    max_new_tokens=320,
    top_p=0.95,
    # Sized for the M5 Pro: generation (group_size * max_new_tokens) dominates and
    # ran ~86 s/step here, so 300 steps lands around an 8-hour overnight run. The
    # bottleneck is MPS generation speed, not memory (per-step cache-free keeps it
    # comfortably inside 48GB). Resume from a checkpoint to extend further.
    total_steps=300,
    save_dir="runs/gsm8k_0p5b",
)

# A bigger run once the 0.5B baseline is beaten. LoRA keeps optimizer state small.
GSM8K_1P5B = Config(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    use_lora=True,
    group_size=8,
    prompts_per_step=2,
    max_new_tokens=640,
    total_steps=1000,
    save_dir="runs/gsm8k_1p5b",
)

PRESETS = {
    "smoke": SMOKE,
    "gsm8k_0p5b": GSM8K_0P5B,
    "gsm8k_1p5b": GSM8K_1P5B,
}


def load(preset: str, **overrides) -> Config:
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}, choose from {sorted(PRESETS)}")
    cfg = PRESETS[preset]
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    return cfg
