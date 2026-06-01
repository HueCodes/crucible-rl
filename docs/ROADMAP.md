# Roadmap

Ordered, not scheduled.

## Phase 0: scaffold (done)
- Project layout, config presets, GSM8K loading, verifiable reward, GRPO core,
  training loop, eval, smoke config.
- Verifier and GRPO math covered by fast tests with no model download.

## Phase 1: beat the 0.5B baseline
- Run `gsm8k_0p5b` end to end on the 48GB machine via MPS.
- Confirm eval accuracy rises above the recorded baseline.
- Save the reward curve and before/after accuracy.

## Phase 2: scale up
- Move to `gsm8k_1p5b` with LoRA.
- Longer completions, tune temperature and group size.
- Watch KL: if it spikes, raise beta or lower lr.

## Phase 3: add code reasoning
- Second reward domain: generated Python scored by unit tests passing.
- Run the candidate code in a sandboxed subprocess with a timeout, reward = fraction of tests passed.
- Mix math and code prompts in a step.

## Phase 4: ablations and writeup
- Sweep beta_kl, group_size, and the format reward weight.
- Plot reward and eval accuracy across the sweep.
- Short writeup of what moved the needle.

## Phase 5: stretch
- Start from a base (non-instruct) model to reproduce the R1-Zero "reasoning emerges" effect.
- Try a learned verifier and compare against the programmatic one.
- Export the best policy for fast local inference.
