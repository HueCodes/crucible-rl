#!/usr/bin/env bash
# Auto-resuming supervisor for the GRPO run.
#
# PyTorch's MPS allocator leaks ~1.5GB/step and won't release it, so a single
# process OOMs after ~25-30 steps. The trainer's memory watchdog checkpoints and
# exits with code 42 just before the ceiling; this loop restarts it with --resume,
# and a fresh process resets the leak. Net effect: the full run completes at full
# quality, just spread across several short-lived processes.
#
#   Usage:  scripts/supervise.sh [preset]    (default: gsm8k_0p5b)
#   Watch:  tail -f runs/<preset>/stdout.log
#
# Exit codes from the trainer: 0 = run complete, 42 = restart for memory,
# anything else = a genuine crash (retried from the last checkpoint a few times).

set -u
PRESET="${1:-gsm8k_0p5b}"
PY=".venv/bin/python"
RUN_DIR="runs/${PRESET}"
SLOG="${RUN_DIR}/supervise.log"
mkdir -p "$RUN_DIR"

attempt=0
hard_fails=0
while true; do
  attempt=$((attempt + 1))
  echo "[supervise] attempt ${attempt} starting $(date)" | tee -a "$SLOG"
  PYTHONUNBUFFERED=1 "$PY" -m crucible.train --preset "$PRESET" --resume \
      >> "${RUN_DIR}/stdout.log" 2>&1
  code=$?
  echo "[supervise] trainer exited code=${code} $(date)" | tee -a "$SLOG"

  if [ "$code" -eq 0 ]; then
    echo "[supervise] run complete after ${attempt} attempt(s)." | tee -a "$SLOG"
    break
  elif [ "$code" -eq 42 ]; then
    hard_fails=0          # clean memory restart = progress was made
    continue
  else
    hard_fails=$((hard_fails + 1))
    if [ "$hard_fails" -ge 5 ]; then
      echo "[supervise] ${hard_fails} consecutive hard failures — stopping." | tee -a "$SLOG"
      exit 1
    fi
    echo "[supervise] hard failure (${hard_fails}/5) — resuming from last checkpoint." | tee -a "$SLOG"
    sleep 5
    continue
  fi
done
