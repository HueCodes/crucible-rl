from __future__ import annotations

import re
from dataclasses import dataclass

from datasets import load_dataset

# DeepSeek R1-Zero style system prompt: think first, then answer, both tagged.
# The format is what the format_reward keys on, and the <answer> span is what the
# verifier parses, so keep it in sync with rewards.extract_answer.
SYSTEM_PROMPT = (
    "You are a careful math problem solver. Think step by step inside "
    "<think> </think>, then give only the final answer inside <answer> </answer>. "
    "The final answer must be a single number."
)

_GOLD_RE = re.compile(r"####\s*([-0-9,\.]+)")


@dataclass
class Sample:
    question: str
    gold: str  # canonical numeric string, e.g. "42"


def _parse_gold(answer_field: str) -> str:
    m = _GOLD_RE.search(answer_field)
    if not m:
        raise ValueError(f"no gold answer found in: {answer_field!r}")
    return m.group(1).replace(",", "").strip()


def load_gsm8k(name: str, config: str, split: str) -> list[Sample]:
    ds = load_dataset(name, config, split=split)
    return [Sample(question=r["question"], gold=_parse_gold(r["answer"])) for r in ds]


def build_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
