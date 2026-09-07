from __future__ import annotations

import re
from typing import Any

SCORE_KEYS = ("exact_match", "rouge_l", "token_f1", "gold_recall", "lexical_hit")

_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    lowered = (text or "").casefold()
    stripped = _PUNCT_RE.sub(" ", lowered)
    return _SPACE_RE.sub(" ", stripped).strip()


def tokenize_answer(text: str) -> list[str]:
    return normalize_answer(text).split()


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def _f1(precision: float, recall: float) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def score_answer(prediction: str, gold: str) -> dict[str, Any]:
    pred_tokens = tokenize_answer(prediction)
    gold_tokens = tokenize_answer(gold)
    if not pred_tokens or not gold_tokens:
        return {
            "exact_match": 0.0,
            "rouge_l": 0.0,
            "token_f1": 0.0,
            "gold_recall": 0.0,
            "lexical_hit": 0.0,
        }

    exact_match = 1.0 if pred_tokens == gold_tokens else 0.0
    token_precision = len(set(pred_tokens) & set(gold_tokens)) / len(set(pred_tokens))
    token_recall = len(set(pred_tokens) & set(gold_tokens)) / len(set(gold_tokens))
    gold_recall = token_recall
    lcs = _lcs_length(pred_tokens, gold_tokens)
    rouge_precision = lcs / len(pred_tokens)
    rouge_recall = lcs / len(gold_tokens)
    rouge_l = _f1(rouge_precision, rouge_recall)
    token_f1 = _f1(token_precision, token_recall)
    lexical_hit = 1.0 if gold_recall >= 0.6 or rouge_l >= 0.5 else 0.0
    return {
        "exact_match": exact_match,
        "rouge_l": rouge_l,
        "token_f1": token_f1,
        "gold_recall": gold_recall,
        "lexical_hit": lexical_hit,
    }
