# Phase 1: beat the 0.5B baseline

The 0.5B GSM8K run on the M5 Pro (48GB, MPS).

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Preset: `gsm8k_0p5b` (fp32, `group_size=4`, `max_new_tokens=256`, 400 steps)
- Reward: a verifier (parse `<answer>`, compare to gold) plus a small format bonus.
  No learned reward model.

## Result

| metric | value |
|--------|-------|
| baseline eval acc (step 0) | **0.125** |
| final eval acc (step 400)  | **0.438** |
| best eval acc              | 0.438 |
| relative improvement       | **3.5×** |

Eval is greedy exact-match over 64 held-out GSM8K test prompts. Gains were largest
in the first ~100 steps. The policy stayed stable the whole run: KL against the
frozen reference peaked at 15.6 and mostly sat near zero, never approaching the
divergence guard at 40. See `docs/training_curves.png`.

## Honesty notes

- The 0.125 baseline *understates* the base model. We score a strict
  `<think>…</think><answer>…</answer>` format, greedy, 0-shot, with a 256-token cap,
  so part of the early gain is the model learning to emit the format the verifier
  expects, not purely new reasoning ability.
- 64-sample eval has ~±6% resolution; the step-to-step wobble (0.33–0.44) is mostly
  noise, not regression.
- The run completed across 25 short-lived processes: PyTorch's MPS allocator leaks
  ~1.5GB/step and OOMs ~step 28, so a memory watchdog checkpoints and `scripts/supervise.sh`
  auto-resumes a fresh process. See `docs/MPS_LEAK.md`.

Phase 1 is done: eval accuracy climbed well above baseline and the curve is saved.
Natural next step (Phase 2): `gsm8k_1p5b` with LoRA for a higher absolute number.
