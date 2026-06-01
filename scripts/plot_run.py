"""Plot a run's reward curve and eval accuracy from metrics.jsonl.

Usage:  uv run python scripts/plot_run.py runs/gsm8k_0p5b

Writes reward_curve.png next to metrics.jsonl and prints baseline and final
eval accuracy to stdout.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(path: str):
    train, eval_ = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("kind") == "train":
                train.append(rec)
            elif rec.get("kind") == "eval":
                eval_.append(rec)
    return train, eval_


def main():
    if len(sys.argv) != 2:
        print("usage: plot_run.py <run_dir>")
        sys.exit(1)
    run_dir = sys.argv[1]
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    train, eval_ = load_metrics(metrics_path)

    fig, ax_reward = plt.subplots(figsize=(8, 5))
    ax_acc = ax_reward.twinx()

    if train:
        ax_reward.plot(
            [r["step"] for r in train],
            [r["reward"] for r in train],
            color="tab:blue",
            alpha=0.4,
            linewidth=1.0,
            label="train reward",
        )
    ax_reward.set_xlabel("step")
    ax_reward.set_ylabel("training reward", color="tab:blue")
    ax_reward.tick_params(axis="y", labelcolor="tab:blue")

    if eval_:
        ax_acc.plot(
            [r["step"] for r in eval_],
            [r["acc"] for r in eval_],
            color="tab:red",
            marker="o",
            linewidth=1.5,
            label="eval acc",
        )
    ax_acc.set_ylabel("eval accuracy", color="tab:red")
    ax_acc.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    out_path = os.path.join(run_dir, "reward_curve.png")
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")

    if eval_:
        print(f"baseline eval acc: {eval_[0]['acc']:.3f}")
        print(f"final eval acc:    {eval_[-1]['acc']:.3f}")
    else:
        print("no eval lines found")


if __name__ == "__main__":
    main()
