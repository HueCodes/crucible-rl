from __future__ import annotations

import re
from dataclasses import dataclass

# Rewards a think-then-answer structure. Searched (not full-matched) so trailing
# or surrounding text doesn't void the bonus — the old anchored ^...$ match meant
# any stray token after </answer> scored zero, so the format reward never fired.
_FORMAT_RE = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
# Pulls the last <answer> span (models sometimes emit more than one).
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# A signed number, integer or decimal, possibly with thousands commas.
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text: str) -> str | None:
    spans = _ANSWER_RE.findall(text)
    region = spans[-1] if spans else text
    nums = _NUMBER_RE.findall(region)
    if not nums:
        return None
    return nums[-1].replace(",", "").strip()


def _to_float(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def correctness_reward(completion: str, gold: str, weight: float) -> float:
    pred = extract_answer(completion)
    if pred is None:
        return 0.0
    p, g = _to_float(pred), _to_float(gold)
    if p is None or g is None:
        return weight if pred.strip() == gold.strip() else 0.0
    return weight if abs(p - g) < 1e-6 else 0.0


def format_reward(completion: str, weight: float) -> float:
    return weight if _FORMAT_RE.search(completion) else 0.0


@dataclass
class RewardBreakdown:
    total: float
    correct: float
    format: float


def score(completion: str, gold: str, correct_w: float, format_w: float) -> RewardBreakdown:
    c = correctness_reward(completion, gold, correct_w)
    f = format_reward(completion, format_w)
    return RewardBreakdown(total=c + f, correct=c, format=f)
