from __future__ import annotations

import json
import os


class JsonlLogger:
    """Append one JSON line per call to save_dir/metrics.jsonl.

    Each line is {step, kind, ...fields}. kind is "train" or "eval".
    Plain JSONL, no extra deps, no timestamps.
    """

    def __init__(self, save_dir: str):
        self._fh = open(os.path.join(save_dir, "metrics.jsonl"), "a", buffering=1)

    def log(self, step: int, kind: str, **fields):
        record = {"step": step, "kind": kind, **fields}
        self._fh.write(json.dumps(record) + "\n")
