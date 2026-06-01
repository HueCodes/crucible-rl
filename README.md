# Crucible

GRPO reasoning trainer. Take a small instruct model and improve its math
reasoning with reinforcement learning against a verifiable reward, the
"R1-Zero on a laptop" recipe, sized to run on a 48GB Mac via MPS.

The reward is a function, not a learned reward model: parse the model's final
answer and check it against the gold answer. That keeps runs reproducible and
ungameable. GSM8K (grade-school math) is the starting domain.

See `docs/NOTES.md` for the GRPO math and `docs/ROADMAP.md` for where this goes.

## Layout

    src/crucible/
      config.py    presets (smoke, gsm8k_0p5b, gsm8k_1p5b) and CLI overrides
      data.py      GSM8K loading, prompt template, gold-answer parsing
      rewards.py   answer extraction, correctness + format rewards (verifier)
      model.py     load policy + frozen reference, group sampling
      grpo.py      group-relative advantage, clipped objective, k3 KL
      train.py     rollout, loss, optimizer loop, periodic eval
      eval.py      greedy exact-match accuracy
    tests/         verifier and GRPO-math checks, no model download
    docs/          NOTES (the math), ROADMAP

## Setup

    uv sync                 # core deps (torch, transformers, datasets)
    uv run pytest           # verify the verifier and GRPO math, fast, no model

## Run

    # tiny end-to-end wiring check on a 0.5B model
    uv run python -m crucible.train --preset smoke

    # the real starter run
    uv run python -m crucible.train --preset gsm8k_0p5b

The smoke run downloads Qwen2.5-0.5B-Instruct and GSM8K on first use, then runs
two steps so you can confirm the loop trains before committing to a full run.
`train.py` prints a baseline eval accuracy at step 0, then reward and eval
accuracy as it goes.

## Notes

- Optional LoRA for the 1.5B preset: `uv sync --extra lora`.
- MPS fallback for unsupported ops is enabled in `train.py`.
- This is standalone. Hardware deployment and an inference engine are separate
  projects, not dependencies.
